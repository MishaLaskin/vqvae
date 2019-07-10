from dm_control import suite
import numpy as np
from gym.spaces import Box, Dict
import os
import torch
from vqvae.models.vqvae import VQVAE
import time
from sys import getsizeof


class DMPointMassEnv:

    def __init__(self,
                 env_name='point_mass',
                 reward_type='dense',
                 indicator_threshold=0.06,
                 mode='easy_big',
                 max_steps=100,
                 **kwargs
                 ):

        self.env_name = env_name
        self.mode = mode
        self.dm_env = suite.load(self.env_name, self.mode)

        self.max_steps = max_steps
        self.indicator_threshold = indicator_threshold
        self.reward_type = reward_type
        self.action_spec = self.dm_env.action_spec()
        self.obs_spec = self.dm_env.observation_spec()
        self.action_space = Box(self.action_spec.minimum,
                                self.action_spec.maximum, dtype=np.float32)
        self.observation_space = Box(low=-5.0, high=5.0, shape=(
            np.sum([x.shape for x in self.obs_spec.values()]),), dtype=np.float32)

        self.episode_success = []

    def _distance_to_target(self):
        return self.dm_env.physics.mass_to_target_dist()

    def _time_step_to_obs(self, ts):
        state = np.concatenate([s for s in ts.observation.values()])
        return state

    def _time_step_to_s_r_d_info(self, action, ts, debug=False):
        state = np.concatenate([s for s in ts.observation.values()])
        reward = self.compute_reward(action, state)
        done = False
        info = {}
        info['dm_reward'] = ts.reward
        info['discount'] = ts.discount
        info['step_type'] = ts.step_type
        info['is_success'] = 0.0

        distance = self._distance_to_target()
        info['distance_to_target'] = distance

        self.returns += reward
        if self.env_step >= self.max_steps:
            done = True
        if distance < self.indicator_threshold:
            done = True
            info['is_success'] = self.env_step

        if done and debug:
            print('ENV STEP', self.env_step, 'start distance', self.start_distance,
                  'end_distance', distance, 'success', info['is_success'], 'returns', self.returns)

        return state, reward, done, info

    def compute_reward(self, action, obs, *args, **kwargs):
        distance_to_target = self._distance_to_target()
        if self.reward_type == 'dense':
            return -distance_to_target
        elif self.reward_type == 'sparse':
            return 0.0 if distance_to_target < self.indicator_threshold else -1.0
        else:
            raise NotImplementedError(
                'Reward type - ' + self.reward_type + ' - not specified')

    def reset(self):
        self.env_step = 0
        self.returns = 0
        self.start_distance = self._distance_to_target()
        time_step = self.dm_env.reset()
        state = self._time_step_to_obs(time_step)
        return state

    def step(self, action):
        assert action.shape == self.action_spec.shape, 'Action must have shape ' + \
            str(action_spec.shape)

        self.env_step += 1
        time_step = self.dm_env.step(action)
        return self._time_step_to_s_r_d_info(action, time_step)

    def render(self, w=128, h=128, camera_id=0):
        return self.dm_env.physics.render(w, h, camera_id=camera_id)

    def _generate_goal_img(self, w=128, h=128, camera_id=0, debug=False):
        env = suite.load(self.env_name, self.mode)

        env.reset()
        while env.physics.mass_to_target_dist() > .1:
            env.reset()
            start_img = env.physics.render(w, h, camera_id=camera_id)

        while env.physics.mass_to_target_dist() > self.indicator_threshold:
            a = self.action_space.sample()
            env.step(a)

        goal_img = env.physics.render(w, h, camera_id=camera_id)
        if debug:
            return start_img, goal_img
        else:
            return goal_img


class DMGoalPointMassEnv(DMPointMassEnv):

    def __init__(self, reward_type='sparse', **kwargs):

        DMPointMassEnv.__init__(self, reward_type=reward_type, **kwargs)
        self.flat_obs_space = Box(low=-5.0, high=5.0, shape=(
            np.sum([x.shape for x in self.obs_spec.values()]),), dtype=np.float32)
        self.goal_space = Box(low=-5.0, high=5.0, shape=(3,), dtype=np.float32)

        self.observation_space = Dict([
            ('observation', self.flat_obs_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.goal_space),
            ('state_observation', self.flat_obs_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.goal_space),
        ])

    def _obs_to_goal_obs(self, obs):
        desired_goal = self.dm_env.physics.named.data.geom_xpos['target']
        achieved_goal = self.dm_env.physics.named.data.geom_xpos['pointmass']
        goal_obs = {
            "observation": obs,
            "desired_goal": desired_goal,
            "achieved_goal": achieved_goal,
            "state_observation": obs,
            "state_desired_goal": desired_goal,
            "state_achieved_goal": achieved_goal,
        }
        return goal_obs

    def step(self, action):
        obs_next, reward, done, info = super().step(action)
        goal_obs_next = self._obs_to_goal_obs(obs_next)
        return goal_obs_next, reward, done, info

    def reset(self):
        obs = super().reset()
        goal_obs = self._obs_to_goal_obs(obs)
        return goal_obs


class DMImageGoalPointMassEnv(DMPointMassEnv):

    def __init__(self,
                 reward_type='sparse',
                 img_dim=32,
                 num_channels=3,
                 fixed_goal=True,
                 **kwargs):

        DMPointMassEnv.__init__(self, reward_type=reward_type, **kwargs)
        self.img_dim = img_dim
        self.num_channels = num_channels

        self.fixed_goal = fixed_goal

        self.flat_obs_space = Box(low=-5.0, high=5.0, shape=(
            np.sum([x.shape for x in self.obs_spec.values()]),), dtype=np.float32)
        self.goal_space = Box(low=-5.0, high=5.0, shape=(3,), dtype=np.float32)
        self.image_space = Box(low=0, high=255, shape=(
            self.img_dim, self.img_dim, self.num_channels))

        self.observation_space = Dict([
            ('observation', self.flat_obs_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.goal_space),
            ('state_observation', self.flat_obs_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.goal_space),
            ('image_desired_goal', self.image_space),
            ('image_achieved_goal', self.image_space),
        ])

    def _obs_to_goal_obs(self, obs):
        achieved_goal, desired_goal = self._get_pointmass_and_target_pos()
        achieved_goal_image = self._get_achived_goal_img()

        goal_obs = {
            "observation": obs,
            "desired_goal": desired_goal,
            "achieved_goal": achieved_goal,
            "state_observation": obs,
            "state_desired_goal": desired_goal,
            "state_achieved_goal": achieved_goal,
            "image_desired_goal": self.desired_goal_image,
            "image_achieved_goal": achieved_goal_image,
        }
        return goal_obs

    def _get_pointmass_and_target_pos(self):
        target_pos = self.dm_env.physics.named.data.geom_xpos['target']
        pm_pos = self.dm_env.physics.named.data.geom_xpos['pointmass']
        return pm_pos, target_pos

    def step(self, action, debug=False):
        if debug:
            start = time.time()

        obs_next, reward, done, info = super().step(action)
        goal_obs_next = self._obs_to_goal_obs(obs_next)

        if debug:
            print('Time per step', time.time()-start)
            print('Size of obs obj', getsizeof(goal_obs_next))

        return goal_obs_next, reward, done, info

    def reset(self):
        if self.fixed_goal:
            # resets env, sets env to goal state, and gets image
            self.desired_goal_image = self._set_fixed_goal_img()
        else:
            self.desired_goal_image = self._set_random_goal_img()
        # resets again to overwrite current state (which is the goal state)
        obs = super().reset()
        goal_obs = self._obs_to_goal_obs(obs)
        return goal_obs

    def _get_achived_goal_img(self):
        return super().render(self.img_dim, self.img_dim)

    def _set_fixed_goal_img(self):

        super().reset()
        _, desired_goal = self._get_pointmass_and_target_pos()
        # some small amount of noise to generate realistic goal
        # scenarios
        noise = np.random.randn(3)*0.015
        noise[-1] = 0.0
        # sets goal state
        self.dm_env.physics.named.data.geom_xpos['pointmass'] = desired_goal + noise
        goal_img = self.render(self.img_dim, self.img_dim)

        return goal_img

    def _set_random_goal_img(self):
        raise NotImplementedError('Only fixed goals have been implemented')


if __name__ == "__main__":
    env_type = 'goal'
    if env_type == 'normal':
        env = DMPointMassEnv()
        obs = env.reset()
        print('reset obs', obs)
        obs_, r, d, info = env.step(env.action_space.sample())
        print('next obs', obs_)
        print('reward', r)
        print('distance to target', env._distance_to_target())
        print('img shape', env.render().shape)
    else:
        env = DMGoalPointMassEnv()
        obs = env.reset()
        print('reset obs', obs)
        obs_, r, d, info = env.step(env.action_space.sample())
        print('next obs', obs_)
        print('reward', r)
        print('distance to target', env._distance_to_target())
        print('img shape', env.render().shape)
