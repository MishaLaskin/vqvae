from dm_control import suite
from gym.spaces import Box, Dict
import numpy as np


# BASE class
class BaseEnv:

    def __init__(self,
                 env_name, mode,
                 obs_dim=None, act_dim=None,
                 **kwargs):

        self.dm_env = suite.load(env_name, mode)
        self.action_space = Box(
            high=1.0, low=-1.0,
            shape=self.dm_env.action_spec().shape if act_dim is None else (act_dim,)
        )

        self.observation_space = Box(
            high=float("inf"), low=-float("inf"),
            shape=(np.sum([v.shape for v in list(self.dm_env.observation_spec().values())]),)) if obs_dim is None else (obs_dim,)

    def reset(self):
        ts = self.dm_env.reset()
        obs = _ts_to_obs(ts)
        return obs

    def step(self, action):
        ts = self.dm_env.step(action)
        obs = _ts_to_obs(ts)
        if _has_attr(self, 'compute_reward'):
            reward, done = self.compute_reward()
        else:
            reward, done = ts.reward, ts.last()
        info = {}
        return obs, reward, done, info


class GoalBaseEnv(BaseEnv):

    def __init__(self,
                 env_name, mode,
                 obs_dim=None, act_dim=None, goal_dim=None,
                 **kwargs):

        BaseEnv.__init__(self, env_name, mode, obs_dim=obs_dim,
                         act_dim=act_dim, **kwargs)\

        if goal_dim is None:
            goal_dim = self.observation_space.shape[0]

        goal_space = Box(
            high=float("inf"), low=-float("inf"),
            shape=(goal_dim,)
        )

        og_obs_space = self.observation_space

        self.observation_space = Dict([
            ('observation', og_obs_space),
            ('desired_goal', goal_space),
            ('achieved_goal', goal_space),
            ('state_observation', og_obs_space),
            ('state_desired_goal', goal_space),
            ('state_achieved_goal', goal_space),
        ])

    def reset(self):
        obs = super().reset()
        obs_dict = _obs_to_ground_truth_dict(obs, self.dm_env.physics)
        return obs_dict

    def step(self):
        obs, reward, done, info = super().step(self)
        obs_dict = _obs_to_ground_truth_dict(obs, self.dm_env.physics)
        return obs_dict, reward, done, info


def _has_attr(obj, name):
    attr = getattr(obj, name, None)
    return True if callable(name) else False


def _ts_to_obs(ts):
    flat_obs = np.array(list(ts.observation.values())).reshape(-1)
    return np.array([v for v in flat_obs])


def _obs_to_ground_truth_dict(obs, physics, object, target):
    desired_goal = physics.named.data.geom_xpos[object]
    achieved_goal = physics.named.data.geom_xpos[target]
    goal_obs = {
        "observation": obs,
        "desired_goal": desired_goal,
        "achieved_goal": achieved_goal,
        "state_observation": obs,
        "state_desired_goal": desired_goal,
        "state_achieved_goal": achieved_goal,
    }
    return goal_obs


if __name__ == "__main__":
    env = BaseEnv('reacher', 'easy')
    obs = env.reset()
    a = env.action_space.sample()
    obs_next, r, d, info = env.step(a)
    print('OBS', obs)
    print('OBS_NEXT', obs_next, 'r', r, 'd', d, 'info', info)

    env = GoalBaseEnv('reacher', 'easy')
    obs = env.reset()
    a = env.action_space.sample()
    obs_next, r, d, info = env.step(a)
    print('OBS', obs)
    print('OBS_NEXT', obs_next, 'r', r, 'd', d, 'info', info)
