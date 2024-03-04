from .base_agent import BaseAgent
from common.misc_util import adjust_lr, adjust_lr_exp, get_n_params
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import wandb
from pprint import pprint
from collections import deque
import matplotlib.pyplot as plt

from common.ct import ent_loss

class ConceptPPO(BaseAgent):
    def __init__(self,
                 env,
                 policy,
                 logger,
                 storage,
                 device,
                 n_checkpoints,
                 n_steps=128,
                 n_envs=8,
                 epoch=3,
                 mini_batch_per_epoch=8,
                 mini_batch_size=32*8,
                 gamma=0.99,
                 lmbda=0.95,
                 learning_rate=2.5e-4,
                 grad_clip_norm=0.5,
                 eps_clip=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 coef_sl=0.5,
                 normalize_adv=True,
                 normalize_rew=True,
                 use_gae=True,
                 **kwargs):
        super().__init__(env, policy, logger, storage, device, n_checkpoints)
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.epoch = epoch
        self.mini_batch_per_epoch = mini_batch_per_epoch
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.lmbda = lmbda
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)
        self.grad_clip_norm = grad_clip_norm
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.normalize_adv = normalize_adv
        self.normalize_rew = normalize_rew
        self.use_gae = use_gae
        self.coef_sl = coef_sl
    
    def predict(self, obs, hidden_state, done):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(device=self.device)
            hidden_state = torch.FloatTensor(hidden_state).to(device=self.device)
            mask = torch.FloatTensor(1-done).to(device=self.device)
            # print("obs into policy: ", obs.shape)
            dist, value, mask_attn, concept_attn = self.policy(obs, hidden_state, mask)
            act = dist.sample()
            log_prob_act = dist.log_prob(act)
            self.expl_pred = (mask_attn.cpu().numpy(), concept_attn.cpu().numpy())

        return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy(), hidden_state.cpu().numpy()

    def optimize(self):
        pi_loss_list, value_loss_list, entropy_loss_list = [], [], []
        sparsity_loss_list, mask_loss_list, concept_loss_list = [], [], []
        ratio_list = []
        rl_loss_list, sl_loss_list = [], []
        batch_size = self.n_steps * self.n_envs // self.mini_batch_per_epoch
        if batch_size < self.mini_batch_size:
            self.mini_batch_size = batch_size
        grad_accumulation_steps = batch_size / self.mini_batch_size
        grad_accumulation_cnt = 1

        self.policy.train()
        for e in range(self.epoch):
            recurrent = self.policy.is_recurrent()
            generator = self.storage.fetch_train_generator(mini_batch_size=self.mini_batch_size,
                                                           recurrent=recurrent)
            for sample in generator:
                obs_batch, patch_mask_batch_label, concept_batch_label, hidden_state_batch, act_batch, done_batch, \
                    old_log_prob_act_batch, old_value_batch, return_batch, adv_batch = sample
                mask_batch = (1-done_batch)
                
                dist_batch, value_batch, patch_mask_batch_pred, concept_batch_pred = self.policy(obs_batch, hidden_state_batch, mask_batch)

                # Sparsity Loss
                sparsity_loss = attn_sparsity_loss(patch_mask_batch_pred, concept_batch_pred)

                # Patch Mask Loss
                mask_loss = patch_mask_loss(patch_mask_batch_pred, patch_mask_batch_label)

                # Concept Loss
                concept_loss = spatial_concept_loss(concept_batch_pred, concept_batch_label)

                # Clipped Surrogate Objective
                log_prob_act_batch = dist_batch.log_prob(act_batch)
                ratio = torch.exp(log_prob_act_batch - old_log_prob_act_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv_batch
                pi_loss = -torch.min(surr1, surr2).mean()

                # Clipped Bellman-Error
                clipped_value_batch = old_value_batch + (value_batch - old_value_batch).clamp(-self.eps_clip, self.eps_clip)
                v_surr1 = (value_batch - return_batch).pow(2)
                v_surr2 = (clipped_value_batch - return_batch).pow(2)
                value_loss = 0.5 * torch.max(v_surr1, v_surr2).mean()

                # Policy Entropy
                entropy_loss = dist_batch.entropy().mean()

                # Total Loss
                rl_loss = pi_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                # sl_loss = sparsity_loss + 0.1 * mask_loss + 0.01 * concept_loss
                sl_loss = 0.01 * concept_loss
                # loss = rl_loss + self.coef_sl * sl_loss
                loss = rl_loss + sl_loss
                loss.backward()

                # Let model to handle the large batch-size with small gpu-memory
                if grad_accumulation_cnt % grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                grad_accumulation_cnt += 1
                pi_loss_list.append(pi_loss.item())
                value_loss_list.append(value_loss.item())
                entropy_loss_list.append(entropy_loss.item())
                sparsity_loss_list.append(sparsity_loss.item())
                mask_loss_list.append(mask_loss.item())
                concept_loss_list.append(concept_loss.item())
                rl_loss_list.append(rl_loss.item())
                sl_loss_list.append(sl_loss.item())
                ratio_list.append(ratio.mean().item())

        summary = {'Loss/pi': np.mean(pi_loss_list),
                   'Loss/v': np.mean(value_loss_list),
                   'Loss/entropy': np.mean(entropy_loss_list),
                   'Loss/sparsity': np.mean(sparsity_loss_list),
                   'Loss/mask': np.mean(mask_loss_list),
                   'Loss/concept': np.mean(concept_loss_list),
                   'Loss/rl': np.mean(rl_loss_list),
                   'Loss/sl': np.mean(sl_loss_list),
                   'Loss/ratio': np.mean(ratio_list)}
        return summary
    
    def train(self):
        info_queue = deque(maxlen=1)
        num_timesteps = 5000000
        save_every = num_timesteps // self.num_checkpoints
        checkpoint_cnt = 0
        obs, info = self.env.reset()
        hidden_state = np.zeros((self.n_envs, self.storage.hidden_state_size))
        done = np.zeros(self.n_envs)
        info_queue.append((info['mask_vector'], info['supervise_signal']))
        while self.t < num_timesteps:
            # Run Policy
            step_return = []
            step_length = []
            self.policy.eval()
            for _ in range(self.n_steps):
                act, log_prob_act, value, next_hidden_state = self.predict(obs, hidden_state, done)
                next_obs, rew, done, truncated, info = self.env.step(act)
                if 'episode' in info.keys():
                    step_return.append(info['episode']['r'][info['_episode']].max())
                    step_length.append(info['episode']['l'][info['_episode']].max())
                self.storage.store(obs, info_queue[0][0], info_queue[0][1], hidden_state, act, rew, done, info, log_prob_act, value)
                obs = next_obs
                hidden_state = next_hidden_state
                info_queue.append((info['mask_vector'], info['supervise_signal']))
            # fig, ax = plt.subplots(4,2)
            # ax[0][0].imshow(info_queue[0][0][0].reshape(21, 16))
            # ax[0][1].imshow(self.expl_pred[0][0].reshape(21, 16))
            # ax[1][0].imshow(info_queue[0][1][0][:,0].reshape(21, 16))
            # ax[1][1].imshow(self.expl_pred[1][0][:,0].reshape(21, 16))
            # ax[2][0].imshow(info_queue[0][1][0][:,1].reshape(21, 16))
            # ax[2][1].imshow(self.expl_pred[1][0][:,1].reshape(21, 16))
            # ax[3][0].imshow(info_queue[0][1][0][:,2].reshape(21, 16))
            # ax[3][1].imshow(self.expl_pred[1][0][:,2].reshape(21, 16))
            # wandb.log({"chart": fig})
            # fig.clf()
            _, _, last_val, hidden_state = self.predict(obs, hidden_state, done)
            self.storage.store_last(obs, hidden_state, last_val)
            # Compute advantage estimates
            self.storage.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)
            # Optimize policy & valueq
            summary = self.optimize()
            # Log the training-procedure
            self.t += self.n_steps * self.n_envs
            summary['timestep'] = self.t
            if len(step_return) > 0:
                summary['return'] = np.array(step_return).max()
                summary['length'] = np.array(step_length).max()
                wandb.log(summary)
                pprint(summary)
            self.optimizer = adjust_lr(self.optimizer, self.learning_rate, self.t, 30000000)
            # self.optimizer = adjust_lr_exp(self.optimizer, init_lr=self.learning_rate, timesteps=self.t, max_timesteps=30000000, decay_rate=0.005)
            # Save the model
            if self.t > ((checkpoint_cnt+1) * save_every):
                torch.save({'state_dict': self.policy.state_dict()}, self.logger.logdir +
                           '/model_no_mask' + str(self.t) + '.pth')
                checkpoint_cnt += 1
        self.env.close()

    def evaluate(self):
        num_timesteps = 10000000
        save_every = num_timesteps // self.num_checkpoints
        checkpoint_cnt = 0
        obs, info = self.env.reset()
        hidden_state = np.zeros((self.n_envs, self.storage.hidden_state_size))
        done = np.zeros(self.n_envs)
        while self.t < num_timesteps:
            # Run Policy
            step_return = []
            step_length = []
            self.policy.eval()
            for _ in range(self.n_steps):
                act, log_prob_act, value, next_hidden_state = self.predict(obs, hidden_state, done)
                next_obs, rew, done, truncated, info = self.env.step(act)
                if 'episode' in info.keys():
                    step_return.append(info['episode']['r'][info['_episode']].max())
                    step_length.append(info['episode']['l'][info['_episode']].max())
                self.storage.store(obs, info["mask_vector"], info["supervise_signal"], hidden_state, act, rew, done, info, log_prob_act, value)
                obs = next_obs
                hidden_state = next_hidden_state
            _, _, last_val, hidden_state = self.predict(obs, hidden_state, done)
            self.storage.store_last(obs, hidden_state, last_val)
            # Compute advantage estimates
            self.storage.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)
            # Optimize policy & valueq
            summary = self.optimize()
            # Log the training-procedure
            self.t += self.n_steps * self.n_envs
            if len(step_return) > 0:
                summary['return'] = np.array(step_return).max()
                summary['length'] = np.array(step_length).max()
                wandb.log(summary)
                pprint(summary)
            self.optimizer = adjust_lr(self.optimizer, self.learning_rate, self.t, 30000000)
        self.env.close()

class PPO(BaseAgent):
    def __init__(self,
                 env,
                 policy,
                 logger,
                 storage,
                 device,
                 n_checkpoints,
                 n_steps=128,
                 n_envs=8,
                 epoch=3,
                 mini_batch_per_epoch=8,
                 mini_batch_size=32*8,
                 gamma=0.99,
                 lmbda=0.95,
                 learning_rate=2.5e-4,
                 grad_clip_norm=0.5,
                 eps_clip=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 normalize_adv=True,
                 normalize_rew=True,
                 use_gae=True,
                 **kwargs):

        super(PPO, self).__init__(env, policy, logger, storage, device, n_checkpoints)

        self.n_steps = n_steps
        self.n_envs = n_envs
        self.epoch = epoch
        self.mini_batch_per_epoch = mini_batch_per_epoch
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.lmbda = lmbda
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)
        self.grad_clip_norm = grad_clip_norm
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.normalize_adv = normalize_adv
        self.normalize_rew = normalize_rew
        self.use_gae = use_gae

    def predict(self, obs, hidden_state, done):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(device=self.device)
            hidden_state = torch.FloatTensor(hidden_state).to(device=self.device)
            mask = torch.FloatTensor(1-done).to(device=self.device)
            dist, value, hidden_state = self.policy(obs, hidden_state, mask)
            act = dist.sample()
            log_prob_act = dist.log_prob(act)

        return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy(), hidden_state.cpu().numpy()
    

    def optimize(self):
        pi_loss_list, value_loss_list, entropy_loss_list = [], [], []
        batch_size = self.n_steps * self.n_envs // self.mini_batch_per_epoch
        if batch_size < self.mini_batch_size:
            self.mini_batch_size = batch_size
        grad_accumulation_steps = batch_size / self.mini_batch_size
        grad_accumulation_cnt = 1

        self.policy.train()
        for e in range(self.epoch):
            recurrent = self.policy.is_recurrent()
            generator = self.storage.fetch_train_generator(mini_batch_size=self.mini_batch_size,
                                                           recurrent=recurrent)
            for sample in generator:
                obs_batch, hidden_state_batch, act_batch, done_batch, \
                    old_log_prob_act_batch, old_value_batch, return_batch, adv_batch = sample
                mask_batch = (1-done_batch)
                dist_batch, value_batch, _ = self.policy(obs_batch, hidden_state_batch, mask_batch)

                # Clipped Surrogate Objective
                log_prob_act_batch = dist_batch.log_prob(act_batch)
                ratio = torch.exp(log_prob_act_batch - old_log_prob_act_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv_batch
                pi_loss = -torch.min(surr1, surr2).mean()

                # Clipped Bellman-Error
                clipped_value_batch = old_value_batch + (value_batch - old_value_batch).clamp(-self.eps_clip, self.eps_clip)
                v_surr1 = (value_batch - return_batch).pow(2)
                v_surr2 = (clipped_value_batch - return_batch).pow(2)
                value_loss = 0.5 * torch.max(v_surr1, v_surr2).mean()

                # Policy Entropy
                entropy_loss = dist_batch.entropy().mean()
                loss = pi_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                loss.backward()

                # Let model to handle the large batch-size with small gpu-memory
                if grad_accumulation_cnt % grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                grad_accumulation_cnt += 1
                pi_loss_list.append(pi_loss.item())
                value_loss_list.append(value_loss.item())
                entropy_loss_list.append(entropy_loss.item())

        summary = {'Loss/pi': np.mean(pi_loss_list),
                   'Loss/v': np.mean(value_loss_list),
                   'Loss/entropy': np.mean(entropy_loss_list)}
        return summary

    def train(self):
        num_timesteps = 10000000
        save_every = num_timesteps // self.num_checkpoints
        checkpoint_cnt = 0
        obs, info = self.env.reset()
        obs = obs.transpose(0, 3, 1, 2)
        hidden_state = np.zeros((self.n_envs, self.storage.hidden_state_size))
        done = np.zeros(self.n_envs)

        while self.t < num_timesteps:
            step_return = []
            step_length = []
            # Run Policy
            self.policy.eval()
            for _ in range(self.n_steps):
                act, log_prob_act, value, next_hidden_state = self.predict(obs, hidden_state, done)
                next_obs, rew, done, truncated, info = self.env.step(act)
                if 'episode' in info.keys():
                    step_return.append(info['episode']['r'][info['_episode']].max())
                    step_length.append(info['episode']['l'][info['_episode']].max())
                next_obs = next_obs.transpose(0, 3, 1, 2)
                self.storage.store(obs, hidden_state, act, rew, done, info, log_prob_act, value)
                obs = next_obs
                hidden_state = next_hidden_state
            _, _, last_val, hidden_state = self.predict(obs, hidden_state, done)
            self.storage.store_last(obs, hidden_state, last_val)
            # Compute advantage estimates
            self.storage.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)

            # Optimize policy & valueq
            summary = self.optimize()
            # Log the training-procedure
            self.t += self.n_steps * self.n_envs
            if len(step_return) > 0:
                summary['return'] = np.array(step_return).max()
                summary['length'] = np.array(step_length).max()
                wandb.log(summary)
                pprint(summary)
            self.optimizer = adjust_lr(self.optimizer, self.learning_rate, self.t, 30000000)
            # Save the model
            if self.t > ((checkpoint_cnt+1) * save_every):
                torch.save({'state_dict': self.policy.state_dict()}, self.logger.logdir +
                           '/model_' + str(self.t) + '.pth')
                checkpoint_cnt += 1
        self.env.close()

def attn_sparsity_loss(mask_vec, concept_attn):
    """
        Penalty of high entropy output
    """
    cost = ent_loss(mask_vec) if mask_vec is not None else 0.0
    if concept_attn is not None:
        cost = cost + ent_loss(concept_attn)
    return cost

def patch_mask_loss(mask_vec, mask_targets):
    """
        Frobenius norm.
        Args:
            - mask_vec: torch.Tensor of size (batch_size, num_patches, 1)
            - mask_targets: torch.Tensor of size(batch_size, num_patches, 1)
        Using ``torch.nn.BCEWithLogitsLoss`` to compute the loss 
    """
    loss = F.mse_loss(mask_vec, mask_targets, reduction="mean")
    return loss * 21 * 16

def spatial_concept_loss(concept_attn, concept_targets):
    """
        Frobenius norm.
        Args:
            - concept_attn: torch.Tensor of size (batch_size, num_patches, num_concepts)
            - concept_targets: torch.Tensor of size(batch_size, num_patches, num_concepts)
    """
    if concept_attn is None:
        return 0.0
    norm = concept_targets.sum(-1, keepdims=True)
    idx = ~torch.isnan(norm).squeeze()
    if not torch.any(idx):
        return 0.0
    norm_concept_targets = (concept_targets[idx] / (norm[idx] + 1e-8)).float()
    n_concepts = norm_concept_targets.shape[-1]
    return n_concepts * 21 * 16 * F.mse_loss(concept_attn[idx], norm_concept_targets, reduction="mean")

def alpha_blending(original_image, attention_map):
    pass

