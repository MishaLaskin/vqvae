from .utils import GoalBaseEnv, BaseEnv, L2_norm, GoalVQVAEEnv
import numpy as np
from .utils import _ts_to_obs


class EasyPusher(BaseEnv):
    def __init__(self,
                 goal_name='reach_block',
                 max_steps=1000,
                 threshold=7e-2):
        self.max_steps = max_steps
        self.threshold = threshold
        self.goal_name = goal_name
        super().__init__('stacker', 'stack_1')
        self.steps = 0

    def reset(self):
        obs = stationary_reset(self.dm_env)
        return obs

    def compute_reward(self, action, obs):
        r = -self._distance_finger_to_block()
        if self.goal_name == 'push_block':
            r += -self._distance_block_to_target()
        return r

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
        if self.goal_name == 'reach_block':
            distance = self._distance_finger_to_block()
        elif self.goal_name == 'push_block':
            distance = self._distance_block_to_target()
        return distance < self.threshold

    def is_done(self):
        return self.is_success() or self.steps >= self.max_steps

    def _distance_block_to_target(self):
        x = self.dm_env.physics.named.data.geom_xpos
        return np.linalg.norm(x['box0'] - x['target'])

    def _distance_finger_to_block(self):
        x = self.dm_env.physics.named.data.geom_xpos
        d1 = np.linalg.norm(x['box0'] - x['finger1'])
        d2 = np.linalg.norm(x['box0'] - x['finger2'])
        # return shorter of two distances
        return d1 if d1 < d2 else d2


class GoalPusher(GoalBaseEnv):
    def __init__(self,
                 max_steps=1000,
                 threshold=0.05,
                 reward_type='sparse',
                 img_dim=32,
                 camera_id=0
                 ):

        super().__init__(env_name='stacker', mode='stack_1', goal_dim=3)
        self.threshold = threshold
        self.max_steps = max_steps
        self.steps = 0
        self.reward_type = reward_type
        self.desired_goal = None
        self.img_dim = img_dim
        self.camera_id = camera_id

    def compute_reward(self, action, obs, *args, **kwargs):
        distance = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        if self.reward_type == 'sparse':
            r = -1.0 if distance > self.threshold else 0.0
        elif self.reward_type == 'dense':
            r = -distance
        else:
            raise_error(err_type='reward')
        return r

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['achieved_goal']
        desired_goals = obs['desired_goal']
        distances = np.linalg.norm(achieved_goals - desired_goals, axis=1)
        if self.reward_type == 'sparse':
            r = -(distances > self.threshold).astype(float)
        elif self.reward_type == 'dense':
            r = - distances.astype(float)
        else:
            raise_error(err_type='reward')
        return r

    def is_done(self, obs):
        # check if max step limit is reached
        if self.steps >= self.max_steps:
            done = True
            is_success = False
            return done, is_success

        # check if episode was successful
        distance = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])
        is_success = distance < self.threshold
        done = is_success

        return done, is_success

    def step_and_get_obs_dict(self, action):
        assert self.desired_goal is not None, "Must set desired goal before stepping"
        ts = self.dm_env.step(action)
        flat_obs = _ts_to_obs(ts)
        achieved_goal = self.get_achieved_goal()
        obs_dict = self.construct_obs_dict(
            flat_obs, achieved_goal, self.desired_goal)
        return obs_dict

    def reset(self):
        flat_obs = stationary_reset(self.dm_env)
        self.desired_goal = self.get_desired_goal()

        achieved_goal = self.get_achieved_goal()

        obs_dict = self.construct_obs_dict(
            flat_obs, achieved_goal, self.desired_goal)

        self.update_internal_state(reset=True)
        return obs_dict

    def update_internal_state(self, reset=False):
        if reset:
            self.steps = 0
        else:
            self.steps += 1

    def get_desired_goal(self):
        desired_goal = self.dm_env.physics.named.data.geom_xpos['target'].copy(
        )
        return desired_goal

    def get_achieved_goal(self):
        achieved_goal = self.dm_env.physics.named.data.geom_xpos['box0'].copy(
        )
        return achieved_goal

    def construct_obs_dict(self, flat_obs, achieved_goal, desired_goal):
        obs_dict = dict(
            observation=flat_obs,
            state_observation=flat_obs,
            achieved_goal=achieved_goal,
            state_achieved_goal=achieved_goal,
            desired_goal=desired_goal,
            state_desired_goal=desired_goal
        )
        return obs_dict


def rand_a(spec):
    return np.random.uniform(spec.minimum, spec.maximum, spec.shape)


def stationary_reset(env, n_steps=200):

    def try_reset(env):
        env.reset()
        spec = env.action_spec()
        for _ in range(n_steps):
            action = rand_a(spec)
            ts = env.step(action)
        return ts

    ts = try_reset(env)
    xyz = get_xpos(env, name='box0')
    while xyz[2] > 0.05:
        ts = try_reset(env)
        xyz = get_xpos(env, name='box0')

    obs = _ts_to_obs(ts)
    return obs


def get_xpos(env, name=None):
    return env.physics.named.data.geom_xpos[name]


def raise_error(err_type='reward'):
    if err_type == 'reward':
        raise NotImplementedError('Must use dense or sparse reward')
    elif err_type == 'obs_dict':
        raise ValueError('obs must be a dict')
