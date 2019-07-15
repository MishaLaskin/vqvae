from .utils import GoalBaseEnv, BaseEnv, L2_norm
import numpy as np


class EasyReacher(BaseEnv):
    def __init__(self, max_steps=1000, threshold=5e-2):
        self.max_steps = max_steps
        self.threshold = threshold
        super().__init__('reacher', 'easy')
        self.steps = 0

    def compute_reward(self, action, obs):
        return -self._distance_to_target()

    def step(self, action):
        self.steps += 1
        obs, r, d, info = super().step(action)
        r = self.compute_reward(action, obs)
        info['is_success'] = self.is_success()
        d = self.is_done()
        if d:
            self.steps = 0
        return obs, r, d, info

    def is_success(self):
        return self._distance_to_target() < self.threshold

    def is_done(self):
        return self.is_success() or self.steps >= self.max_steps

    def _distance_to_target(self):
        x = self.dm_env.physics.named.data.geom_xpos
        return np.linalg.norm(x['target'] - x['finger'])


class Reacher(GoalBaseEnv):
    def __init__(self,
                 max_steps=1000,
                 threshold=5e-2,
                 max_reward_hits=1,
                 fixed_length=False):

        super().__init__('reacher', 'easy', 'finger', 'target', goal_dim=3)
        self.threshold = threshold
        self.max_steps = max_steps
        self.steps = 0
        self.reward_hits = 0
        self.max_reward_hits = max_reward_hits
        self.fixed_length = fixed_length

    def compute_reward(self, action, obs, *args, **kwargs):
        x = L2_norm(obs['achieved_goal'], obs['desired_goal'])
        r = -1.0 if x > self.threshold else 0.0
        if r == 0.0:
            self.reward_hits += 1
        return r

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['achieved_goal']
        desired_goals = obs['desired_goal']
        distances = np.linalg.norm(achieved_goals - desired_goals, axis=1)
        r = -(distances > self.threshold).astype(float)
        return r

    def step(self, action):
        self.steps += 1
        obs, _, _, info = super().step(action)
        if not isinstance(obs, dict):
            print(obs)
            raise ValueError('obs must be a dict')
        reward = self.compute_reward(action, obs)
        done = self._is_done(reward)

        info = {
            "is_success": self.steps if done and self.reward_hits > 0 else 0.0,
            "reward_hits": self.reward_hits
        }
        return obs, reward, done, info

    def _is_done(self, reward):
        done = False
        if not self.fixed_length:
            if self.reward_hits >= self.max_reward_hits or self.steps >= self.max_steps:
                done = True
        else:
            if self.steps >= self.max_steps:
                done = True
        if done:
            self.steps = 0
            self.reward_hits = 0

        return done

    def reset(self):
        obs = super().reset()
        if not isinstance(obs, dict):
            print(obs)
            raise ValueError('obs must be a dict')
        return obs
