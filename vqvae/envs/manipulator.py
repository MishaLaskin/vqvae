from vqvae.envs.utils import BaseEnv, L2_norm
import numpy as np


class EasyManipulator(BaseEnv):
    def __init__(self, max_steps=1000, threshold=5e-2):
        self.max_steps = max_steps
        self.threshold = threshold
        super().__init__('manipulator', 'bring_ball')

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
        return np.linalg.norm(x['ball'] - x['target_ball'])

    def reset(self):
        obs = super().reset()
        #for _ in range(100):
        #    obs,_,_,_ = super().step(self.action_space.sample())
        return obs


if __name__ == "__main__":
    env = EasyManipulator()
    obs = env.reset()
    print(obs)
    obs = env.step(env.action_space.sample())
    print(obs)