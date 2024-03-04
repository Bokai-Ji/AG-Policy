import numpy as np
import os
import gymnasium as gym
import numpy as np
from skimage.util import view_as_windows
from gymnasium.wrappers import NormalizeObservation
os.environ.setdefault('PATH', '')

class NormObservation(NormalizeObservation):
    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        super().__init__(env, epsilon)

    def step(self, action):
        """Steps through the environment and normalizes the observation."""
        self.obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if self.is_vector_env:
            norm_obs = self.normalize(self.obs)
        else:
            norm_obs = self.normalize(np.array([self.obs]))[0]
        return norm_obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        self.obs, info = self.env.reset(**kwargs)

        if self.is_vector_env:
            return self.normalize(self.obs), info
        else:
            return self.normalize(np.array([self.obs]))[0], info

    def normalize(self, obs):
        """Normalises the observation using the running mean and variance of the observations."""
        self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
    
class AttentionGuider(gym.Wrapper):
    """
    We create a wrapper for some of the Atari environments which the objects can be distinguished by color.
    When you want the agent to focus on a specific object, you can use this wrapper to guide the agent's attention.
    Specifically, the agent will focus on the object that you provide in the concept_dict.
    Attention map provided by this wrapper is pixel-wised precise, and different objects are distinguished by different values (e.g. 0, 1, 2, 3, ...).
    We distinguish objects by color so that we can futher visualize the attention on different objects with various color.
    Thus, you can have a better illustration not only "where" the agent is looking at, but also "what" the agent is looking at.
    We are not likely to use this pixel-wise attention but inflate it to a patch-wise attention to meet the need of feature map produced by CNNs.
    """
    def __init__(self, env, concept_dict=None, search_area=None, patch_size=None) -> None:
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.concept_dict = {
            "player": np.array([92, 186, 92]),
            "enemy": np.array([213, 130, 74]),
            "ball": np.array([236, 236, 236]),
        } if concept_dict is None else concept_dict
        self.search_area = [[34, 194],[]] if search_area is None else search_area # Do not search the score board [34, 194)
        self.patch_size = 5 if patch_size is None else patch_size
        self.attention_map = None

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if self.is_vector_env:
            # print("Is vector")
            infos[0]["supervise_signal"] = self.get_supervise_signal(obs) #! Not yet correct I think
            infos[0]["attention_map"] = self.get_attention_map(obs) #! Not yet correct I think
            infos[0]["mask_vector"] = self.get_mask_vector(obs) #! Not yet correct I think
        else:
            infos["supervise_signal"] = self.get_supervise_signal(np.array(obs))
            infos["attention_map"] = self.get_attention_map(np.array(obs))
            infos["mask_vector"] = self.get_mask_vector(obs)
        return obs, rews, terminateds, truncateds, infos
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.is_vector_env:
            info[0]["supervise_signal"] = self.get_supervise_signal(obs)
            info[0]["attention_map"] = self.get_attention_map(obs) #! Not yet correct I think
            info[0]["mask_vector"] = self.get_mask_vector(obs)
        else:
            info["supervise_signal"] = self.get_supervise_signal(np.array(obs))
            info["attention_map"] = self.get_attention_map(np.array(obs))
            info["mask_vector"] = self.get_mask_vector(obs)
        return obs, info
    
    def get_attention_map(self, obs):
        """Generate attention map from observation. Can be used for visualization."""
        attn_map = np.zeros(obs.shape[:2])
        for idx, concept in enumerate(self.concept_dict.keys()):
            attn_map[self.search_area[0][0]:self.search_area[0][1],:] += np.all(obs == self.concept_dict[concept], axis=-1)[self.search_area[0][0]:self.search_area[0][1],:] * (idx+1)
        return attn_map
    
    def get_supervise_signal(self, obs):
        """Generate supervise signal from observation. Can be used for training."""
        contains_concept = []
        attn_map = self.get_attention_map(obs)
        for idx, concept in enumerate(self.concept_dict.keys()):
            equals_concept = attn_map == idx+1
            patches = view_as_windows(equals_concept, (self.patch_size, self.patch_size), step=self.patch_size)
            contains_concept.append(np.any(patches, axis=(-1,-2)).astype(np.float32).flatten())
        return np.array(contains_concept).transpose(1,0)
    
    def get_mask_vector(self, obs):
        """Generate mask vector from observation. Can be used for training."""
        attn_map = self.get_supervise_signal(obs)
        attn_map = attn_map.sum(-1)
        mask_map = (attn_map > 0).astype(float)
        mask_vector = mask_map.reshape(-1, 1)
        return mask_vector
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    concept_dict_space_invaders = {
        "enemies": np.array([134, 134, 29]),
        "player": np.array([50, 132, 50]),
        "shield": np.array([181, 83, 40]),
        "bullet": np.array([142, 142, 142]),
    }
    search_area_space_invaders = [[31, 195], []]
    env = gym.make('SpaceInvadersNoFrameskip-v4')
    env = AttentionGuider(env, concept_dict=concept_dict_space_invaders, search_area=search_area_space_invaders, patch_size=5)
    env = NormObservation(env)
    obs = env.reset()
    done = False
    truncated = False
    plt.ion()
    plt.figure(dpi=200)
    while not done and not truncated:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        my_obs = obs
        alpha = 0.4
        attn_obs = alpha * my_obs + (1-alpha) * np.concatenate([info["attention_map"].reshape(210, 160, 1)]*3, axis=-1)
        plt.subplot(1, 5, 1)
        plt.imshow(my_obs*50)
        plt.title("Normalized")
        plt.subplot(1, 5, 2)
        plt.imshow(env.obs)
        plt.title("Original")
        plt.axis('off')
        plt.subplot(1, 5, 3)
        plt.imshow(attn_obs)
        plt.title("Combined")
        plt.axis('off')
        plt.subplot(1, 5, 4)
        plt.imshow(info["attention_map"])
        plt.title("Attention")
        plt.axis('off')
        plt.subplot(1, 5, 5)
        plt.imshow(info["supervise_signal"].sum(-1).reshape(42, 32))
        plt.title("Supervise Signal")
        plt.axis('off')
        plt.pause(0.01)
        plt.clf()
    plt.ioff()