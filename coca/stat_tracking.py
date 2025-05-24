import numpy as np
from collections import deque


class PerPromptStatTracker:
    def __init__(self, buffer_size, min_count):
        self.buffer_size = buffer_size
        self.min_count = min_count
        self.stats = {"rewards": {}, "temporal_rewards": {}}

    def update(self, prompts, rewards):
        prompts = np.array(prompts)
        rewards = np.array(rewards)
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards)
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats["rewards"]:
                self.stats["rewards"][prompt] = deque(maxlen=self.buffer_size)
            self.stats["rewards"][prompt].extend(prompt_rewards)

            if len(self.stats["rewards"][prompt]) < self.min_count:
                mean = np.mean(rewards)
                std = np.std(rewards) + 1e-6
            else:
                
                mean = np.mean(self.stats["rewards"][prompt])
                std = np.std(self.stats["rewards"][prompt]) + 1e-6
            advantages[prompts == prompt] = (prompt_rewards - mean) / std

        return advantages
    
    def update_temporal_rewards(self, prompts, temporal_rewards):
        prompts = np.array(prompts)
        temporal_rewards = np.array(temporal_rewards)
        unique = np.unique(prompts)
        final_temporal_rewards = np.empty_like(temporal_rewards)
        for prompt in unique:
            prompt_temporal_rewards = temporal_rewards[prompts == prompt]
            prompt_temporal_rewards_mean = np.mean(prompt_temporal_rewards, axis=1)
            prompt_temporal_rewards_std = np.std(prompt_temporal_rewards, axis=1)
            if prompt not in self.stats["temporal_rewards"]:
                self.stats["temporal_rewards"][prompt] = deque(maxlen=self.buffer_size)
            self.stats["temporal_rewards"][prompt].extend([prompt_temporal_rewards_mean, prompt_temporal_rewards_std])

            if len(self.stats["temporal_rewards"][prompt]) < self.min_count:
                mean = np.mean(temporal_rewards)
                std = np.std(temporal_rewards)
            else:
                mean = np.mean(self.stats["temporal_rewards"][prompt][0])
                std = np.sqrt(
                    np.mean(
                        np.square(self.stats["temporal_rewards"][prompt][0]) + np.square(self.stats["temporal_rewards"][prompt][1])
                    ) - np.square(mean)
                )
            final_temporal_rewards[prompts == prompt] = (prompt_temporal_rewards - mean) / (std + 1e-6)

        return final_temporal_rewards
    
    def get_stats(self):
        return {
            k: {"mean": np.mean(v), "std": np.std(v), "count": len(v)}
            for k, v in self.stats.items()
        }
