import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .misc_util import orthogonal_init

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8,
                 attention_dropout=0.1,
                 projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False) # Extend dim to 3
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape # [BatchSize, NumPairs, Channel]
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads) # [BatchSize, NumPairs, 3, NumHeads, Channel//NumHeads]
            .permute(2, 0, 3, 1, 4) # [3, BatchSize, NumHeads, NumPairs, Channel//NumHeads]
        )
        q, k, v = qkv[0], qkv[1], qkv[2] # Get Q,K,V

        attn = (q @ k.transpose(-2, -1)) * self.scale # S_ = QK^T / sqrt(d)
        attn = attn.softmax(dim=-1) # S = softmax(S_)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # V' = SV
        x = self.proj(x) # FeedForward
        x = self.proj_drop(x)
        return x, attn
    
class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        n_outputs=None,
        attention_dropout=0.0, 
        projection_dropout=0.0
    ):
        super().__init__()
        n_outputs = n_outputs if n_outputs else dim # Classification Classes (Action Space)
        self.num_heads = num_heads                    # Numheads
        self.scale = (dim//num_heads) ** -0.5         # sqrt(head_dim)

        self.q = nn.Linear(dim, dim, bias=False)       # WQ
        self.kv = nn.Linear(dim, dim * 2, bias=False)  # WK, WV
        self.attn_drop = nn.Dropout(attention_dropout)

        self.proj = nn.Linear(dim, n_outputs)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x, y):
        B, Nx, C = x.shape   # shape of Query [BatchSize, NumPatches, EmbeddingLength]
        By, Ny, Cy = y.shape # shape of Key and Value [BatchSize, NumConcepts, EmbeddingLength]

        assert C == Cy, "Feature size of x and y must be the same"  # Assuming same EmbeddingLength

        q = self.q(x).reshape(B, Nx, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # [1, BatchSize, NumHeads, Nx, C//NumHeads]
        kv = (
            self.kv(y)
            .reshape(By, Ny, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q = q[0]                # [BatchSize, NumHeads, NumPatches, EmbeddingLength//NumHeads]
        k, v = kv[0], kv[1]     # [BatchSize, NumHeads, NumConcepts, EmbeddingLength//NumHeads]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nx, C)    # [BatchSize, NumPatches, EmbeddingLength]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn          # x: [BatchSize, NumPatches, ActionDimension], attn: [BatchSize, NumPatches, NumConcepts]
    
class SpatialConceptTransformer(nn.Module):
    def __init__(
        self,
        embedding_dim=768,
        num_actions=10,
        num_heads=2,
        attention_dropout=0.0,
        projection_dropout=0.0,
        n_concepts=10,
        *args,
        **kwargs,):
        super().__init__()
        self.n_concepts = n_concepts
        self.concepts = nn.Embedding(n_concepts, embedding_dim)
        self.patch_mask = SelfAttention(dim = embedding_dim,
                                        num_heads = num_heads,
                                        attention_dropout = attention_dropout,
                                        projection_dropout = projection_dropout)
        self.concept_query = CrossAttention(dim = embedding_dim,
                                            num_heads = num_heads,
                                            n_outputs = num_actions,
                                            attention_dropout = attention_dropout,
                                            projection_dropout = projection_dropout)
        self.cnn_proj = orthogonal_init(nn.Linear(embedding_dim * 21 * 16, embedding_dim))
        self.res_proj = orthogonal_init(nn.Linear(embedding_dim, num_actions), gain=0.01)
    
    def forward(self, x):
        B, Nx, E = x.shape # [BatchSize, NumPatches, EmbeddingLength]
        hidden = x + (self.get_positional_encoding(Nx, E)).cuda() # Sinusoidal Positional Encoding
        mask_attn, concept_attn = None, None
        out_mask, mask_attn = self.patch_mask(hidden)
        mask_vector = mask_attn.mean(dim=1).mean(dim=1).unsqueeze(2) # [BatchSize, NumPatches, 1] Average over heads and rows
        # mask_vector = torch.ones(21*16).cuda()
        # Which is the better way? Mask out original feature or feature after self-attention?
        masked_obs = mask_vector * hidden # [BatchSize, NumPatches, EmbeddingLength] Mask out task irrelative patches
        logits, concept_attn = self.concept_query(masked_obs, torch.stack([self.concepts.weight]*B, dim=0))
        # logits, concept_attn = self.concept_query(x, torch.stack([self.concepts.weight]*B, dim=0))
        concept_attn = concept_attn.mean(dim=1) # [BatchSize, NumPatches, NumConcepts] Average over heads
        out = logits.mean(dim=1) # [BatchSize, ActionDimension]
        # 残差
        x = x.reshape(B, -1)
        x = self.cnn_proj(x)
        x = F.relu(x)
        x = self.res_proj(x)
        out = 0.001 * out + x
        # print("shapes in ConceptTransformer:")
        # print(out.shape, mask_vector.shape, concept_attn.shape)
        return out, mask_vector, concept_attn

    def get_positional_encoding(self, N, d_model):
        # max_seq_len: the maximum length of the sequence
        # embed_dim: the dimension of the embedding vector
        # return: a tensor of shape (max_seq_len, embed_dim)
        PE = torch.zeros(N, d_model)
        positions = torch.arange(0, N, dtype=torch.float).unsqueeze(1)
        i = torch.arange(0, d_model, 2, dtype=torch.float)
        div = positions / (10000 ** (i / d_model))
        PE[:, 0::2] = torch.sin(div)
        PE[:, 1::2] = torch.cos(div)
        return PE

    
# Shared methods
def ent_loss(probs):
    """Entropy loss"""
    ent = -probs * torch.log(probs + 1e-8)
    return ent.mean()


def concepts_sparsity_cost(concept_attn, spatial_concept_attn):
    cost = ent_loss(concept_attn) if concept_attn is not None else 0.0
    if spatial_concept_attn is not None:
        cost = cost + ent_loss(spatial_concept_attn)
    return cost


def concepts_cost(concept_attn, attn_targets):
    """Non-spatial concepts cost
        Attention targets are normalized to sum to 1,
        but cost is normalized to be O(1)

    Args:
        attn_targets, torch.tensor of size (batch_size, n_concepts): one-hot attention targets
    """
    if concept_attn is None:
        return 0.0
    if attn_targets.dim() < 3:
        attn_targets = attn_targets.unsqueeze(1)
    norm = attn_targets.sum(-1, keepdims=True)
    idx = ~torch.isnan(norm).squeeze()
    if not torch.any(idx):
        return 0.0
    # MSE requires both floats to be of the same type
    norm_attn_targets = (attn_targets[idx] / norm[idx]).float()
    n_concepts = norm_attn_targets.shape[-1]
    return n_concepts * F.mse_loss(concept_attn[idx], norm_attn_targets, reduction="mean")


def spatial_concepts_cost(spatial_concept_attn, attn_targets):
    """Spatial concepts cost
        Attention targets are normalized to sum to 1

    Args:
        attn_targets, torch.tensor of size (batch_size, n_patches, n_concepts):
            one-hot attention targets

    Note:
        If one patch contains a `np.nan` the whole patch is ignored
    """
    if spatial_concept_attn is None:
        return 0.0
    norm = attn_targets.sum(-1, keepdims=True)
    idx = ~torch.isnan(norm).squeeze()
    if not torch.any(idx):
        return 0.0
    norm_attn_targets = (attn_targets[idx] / norm[idx]).float()
    n_concepts = norm_attn_targets.shape[-1]
    return n_concepts * F.mse_loss(spatial_concept_attn[idx], norm_attn_targets, reduction="mean")
