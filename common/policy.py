from .misc_util import orthogonal_init
from .model import GRU
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from .ct import SpatialConceptTransformer

class CategoricalPolicy(nn.Module):
    def __init__(self, 
                 embedder,
                 recurrent,
                 action_size):
        """
        embedder: (torch.Tensor) model to extract the embedding for observation
        action_size: number of the categorical actions
        """ 
        super(CategoricalPolicy, self).__init__()
        self.embedder = embedder
        # small scale weight-initialization in policy enhances the stability        
        self.fc_policy = orthogonal_init(nn.Linear(self.embedder.output_dim, action_size), gain=0.01)
        self.fc_value = orthogonal_init(nn.Linear(self.embedder.output_dim, 1), gain=1.0)

        self.recurrent = recurrent
        if self.recurrent:
            self.gru = GRU(self.embedder.output_dim, self.embedder.output_dim)

    def is_recurrent(self):
        return self.recurrent

    def forward(self, x, hx, masks):
        hidden = self.embedder(x)
        if self.recurrent:
            hidden, hx = self.gru(hidden, hx, masks)
        logits = self.fc_policy(hidden)
        log_probs = F.log_softmax(logits, dim=1)
        p = Categorical(logits=log_probs)
        v = self.fc_value(hidden).reshape(-1)
        return p, v, hx
    
class ConceptPolicy(nn.Module):
    def __init__(self,
                 embedder,
                 recurrent,
                 action_size,
                 *args,
                 **kwargs) -> None:
        super().__init__()
        """
        embedder: Vision Core
        action_size: number of the categorical actions
        """
        self.embedder = embedder
        self.policy = SpatialConceptTransformer(embedding_dim=self.embedder.output_dim,
                                                num_actions=action_size,
                                                num_heads=kwargs["num_heads"],
                                                n_outputs=action_size,
                                                attention_dropout=0.0,
                                                projection_dropout=0.0,
                                                n_concepts=kwargs["n_concepts"])
        self.critic = orthogonal_init(nn.Linear(self.embedder.output_dim, 1), gain=1.0)
        self.recurrent = recurrent
    
    def is_recurrent(self):
        return self.recurrent

    def forward(self, x, hidden_state=None, masks=None):
        hidden = self.embedder(x)
        # print("hidden into policy: ", hidden.shape)
        logits, mask_attn, concept_attn = self.policy(hidden)
        log_probs = F.log_softmax(logits, dim=1)
        p = Categorical(logits=log_probs)
        hidden = hidden.mean(dim=1)
        v = self.critic(hidden).reshape(-1)
        return p, v, mask_attn, concept_attn
