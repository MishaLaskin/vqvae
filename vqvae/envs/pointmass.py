from .utils import GoalBaseEnv, BaseEnv, L2_norm, VQVAEEnv, GoalVQVAEEnv
import numpy as np
import torch


class EasyPointmass(BaseEnv):
    def __init__(self, max_steps=1000, threshold=5e-2):
        self.max_steps = max_steps
        self.threshold = threshold
        super().__init__('point_mass', 'easy_big')
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
        return np.linalg.norm(x['target'] - x['pointmass'])


class GoalPointmass(GoalBaseEnv):
    def __init__(self,
                 max_steps=1000,
                 threshold=5e-2,
                 reward_type='dense'):

        super().__init__(env_name='point_mass', mode='easy_big', goal_dim=3)
        self.threshold = threshold
        self.max_steps = max_steps
        self.steps = 0
        self.reward_type = reward_type
        self.desired_goal = None

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
        flat_obs = np.array(list(ts.observation.values())).reshape(-1)
        achieved_goal = self.dm_env.physics.named.data.geom_xpos['pointmass']
        obs_dict = self.construct_obs_dict(
            flat_obs, achieved_goal, self.desired_goal)
        return obs_dict

    def reset(self):
        self.desired_goal = self.set_desired_goal()
        obs_dict = self.reset_and_get_obs_dict()
        self.update_internal_state(reset=True)
        return obs_dict

    def update_internal_state(self, reset=False):
        if reset:
            self.steps = 0
        else:
            self.steps += 1

    def set_desired_goal(self):
        self.dm_env.reset()
        desired_goal = self.dm_env.physics.named.data.geom_xpos['target'].copy(
        )
        return desired_goal

    def reset_and_get_obs_dict(self):
        # make sure desired goal is set
        assert self.desired_goal is not None, "Must set desired goal before resetting env again"
        # get current observation
        ts = self.dm_env.reset()
        flat_obs = np.array(list(ts.observation.values())).reshape(-1)
        achieved_goal = self.dm_env.physics.named.data.geom_xpos['pointmass'].copy(
        )
        obs_dict = self.construct_obs_dict(
            flat_obs, achieved_goal, self.desired_goal)
        return obs_dict

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


class EasyPointmassVQVAE(VQVAEEnv):
    def __init__(self,
                 max_steps=1000,
                 threshold=5e-2,
                 model_path='/home/misha/research/vqvae/results/vqvae_temporal_data_long_ne8nd2.pth',
                 rep_type='discrete',
                 **kwargs

                 ):
        self.max_steps = max_steps
        self.threshold = threshold
        super().__init__('point_mass', 'easy_big', model_path=model_path, **kwargs)
        self.steps = 0
        self.visited_reps = set()
        self.rep_type = rep_type

    def reset_goal_image(self):

        super().reset()
        _, desired_goal = self._get_pointmass_and_target_pos()
        # some small amount of noise to generate realistic goal
        # scenarios
        noise = np.random.randn(3)*0.01
        noise[-1] = 0.0
        # sets goal state
        self.dm_env.physics.named.data.geom_xpos['pointmass'] = desired_goal + noise
        goal_img = self.render(
            width=self.img_dim, height=self.img_dim, camera_id=self.camera_id)
        self.goal_img = goal_img
        goal_img = goal_img.reshape(1, *goal_img.shape)

        return goal_img

    def normalize_image(self, img):
        """normalizes image to [-1,1] interval

        Arguments:
            img {np.array or torch.tensor} -- [an image array / tensor with integer values 0-255]

        Returns:
            [np.array or torch tensor] -- [an image array / tensor with float values in [-1,1] interval]
        """
        # takes to [0,1] interval
        img /= 255.0
        # takes to [-0.5,0.5] interval
        img -= 0.5
        # takes to [-1,1] interval
        img /= 0.5
        return img

    def normalize_indices(self, x):
        assert max(x) <= 7.0, 'index mismatch during normalization'
        x = x.float()
        x /= 7.0
        x -= 0.5
        x /= 0.5
        return x

    def reset(self):
        # generate goal, encode it, get continuous and discrete values
        self.encoded_goal, self.goal_indices = self.encode_observation(
            is_goal=True, include_indices=True)
        # reset env
        super().reset()
        # get observation encoding
        z_e, e_indices = self.encode_observation(
            is_goal=False, include_indices=True)

        self.steps = 0
        self.visited_reps = set()
        rep_hash = hash(tuple(e_indices))
        self.visited_reps.add(rep_hash)
        self.current_rep = rep_hash
        self.last_rep = 0

        if self.rep_type == 'discrete':
            return e_indices
        elif self.rep_type == 'continuous':
            return z_e
        elif self.rep_type == 'mixed':
            return np.concatenate((z_e, e_indices))

    def encode_observation(self, is_goal=False, include_indices=False):
        if is_goal:
            img = self.reset_goal_image()
            self.goal_img = img
        else:
            img = self.get_current_image()

        img = self.numpy_to_tensor_img(img)
        img = self.normalize_image(img)


normalize_image
        z_e = self.encode_image(img, as_tensor=True)

        if include_indices:
            _, _, _, _, e_indices = self.model.vector_quantization(z_e)
            e_indices = self.normalize_indices(
                e_indices).detach().cpu().numpy()
            return z_e.reshape(-1).detach().cpu().numpy(), e_indices.reshape(-1)

        return z_e.reshape(-1).detach().cpu().numpy(), None

    def _get_pointmass_and_target_pos(self):
        target_pos = self.dm_env.physics.named.data.geom_xpos['target']
        pm_pos = self.dm_env.physics.named.data.geom_xpos['pointmass']
        return pm_pos, target_pos

    def compute_reward(self, action, z):
        # return 0.0 if self._distance_to_target(z) < self.threshold else -1.0
        # return np.exp(-self._distance_to_target(z))
        x, y = self._get_pointmass_and_target_pos()
        return -np.linalg.norm(x-y)

    def step(self, action):
        self.steps += 1
        self.last_rep = self.current_rep

        # step forward
        _, r, d, info = super().step(action)

        # get latent reps
        z_e, e_indices = self.encode_observation(
            is_goal=False, include_indices=True)

        # compute reward
        r = self.compute_reward(action, z_e)
        info['is_success'] = self.is_success(z_e)
        d = self.is_done(z_e)

        # hash reps
        rep_hash = hash(tuple(e_indices))
        self.visited_reps.add(rep_hash)
        self.current_rep = rep_hash

        if self.rep_type == 'discrete':
            obs_next = e_indices
        elif self.rep_type == 'continuous':
            obs_next = z_e
        elif self.rep_type == 'mixed':
            obs_next = np.concatenate((z_e, e_indices))

        return obs_next, r, d, info

    def is_success(self, z):
        # return self._distance_to_target(z) < self.threshold
        return -self.compute_reward(None, None) < self.threshold

    def is_done(self, z):
        return self.is_success(z) or self.steps >= self.max_steps

    def _distance_to_target(self, z):
        x = self.encoded_goal
        return np.linalg.norm(z - x)


class GoalPointmassVQVAE(GoalVQVAEEnv):
    def __init__(self,
                 obs_dim=None,
                 goal_dim=None,
                 model_path=None,
                 img_dim=32,
                 camera_id=0,
                 gpu_id=0,
                 reward_type='sparse',
                 rep_type='continuous',
                 max_steps=500,
                 threshold=0.05,
                 **kwargs):
        # be sure to specify obs dim and goal dim

        super().__init__(env_name='point_mass', mode='easy_big',
                         obs_dim=obs_dim, act_dim=None, goal_dim=goal_dim,
                         model_path=model_path,
                         img_dim=img_dim,
                         camera_id=camera_id,
                         gpu_id=gpu_id)

        self.threshold = threshold
        self.max_steps = max_steps
        self.steps = 0
        self.reward_type = reward_type
        self.desired_goal = None
        self.rep_type = rep_type
        self.current_rep = None
        self.last_rep = None

    def compute_reward(self, action, obs, *args, **kwargs):
        # abstract method only cares about obs and threshold

        distance = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])

        if self.reward_type == 'sparse':
            r = -1.0 if distance > self.threshold else 0.0
        elif self.reward_type == 'dense':
            r = -distance
        else:
            raise_error(err_type='reward')

        return r

    def compute_rewards(self, actions, obs):
        # abstract method only cares about obs and threshold
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
        # abstract method only cares about obs and threshold
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

    def reset_goal_image(self):

        self.dm_env.reset()
        _, desired_goal = self._get_pointmass_and_target_pos()
        # some small amount of noise to generate realistic goal
        # scenarios
        noise = np.random.randn(3)*0.01
        noise[-1] = 0.0
        # sets goal state
        self.dm_env.physics.named.data.geom_xpos['pointmass'] = desired_goal + noise
        goal_img = self.render(
            width=self.img_dim, height=self.img_dim, camera_id=self.camera_id)
        self.goal_img = goal_img
        goal_img = goal_img.reshape(1, *goal_img.shape)

        return goal_img

    def step(self, a):
        obs_dict = self.step_and_get_obs_dict(a)
        reward = self.compute_reward(a, obs_dict)
        done, is_success = self.is_done(obs_dict)
        info = {
            'is_success': is_success,
            'num_reps': len(self.visited_reps),
            'latent_distance': np.linalg.norm(obs_dict['achieved_goal'] - obs_dict['desired_goal'])
        }

        return obs_dict, reward, done, info

    def step_and_get_obs_dict(self, action):
        assert self.encoded_goal is not None, "Must set desired goal before stepping"

        # 1. step forward with action
        # 2. get image, encode it

        _ = self.dm_env.step(action)

        z_e, e_indices = self.encode_observation(
            is_goal=False, include_indices=True)

        achieved_goal = z_e

        self.update_internal_state(e_indices)
        obs_dict = self.construct_obs_dict(
            z_e, achieved_goal, self.encoded_goal)
        return obs_dict

    def encode_observation(self, is_goal=False, include_indices=False):
        if is_goal:
            img = self.reset_goal_image()
            self.goal_img = img
        else:
            img = self.get_current_image()

        img = self.numpy_to_tensor_img(img)
        img = self.normalize_image(img)

        z_e = self.encode_image(img, as_tensor=True)

        if include_indices:
            _, _, _, _, e_indices = self.model.vector_quantization(z_e)
            e_indices = self.normalize_indices(
                e_indices).detach().cpu().numpy()
            return z_e.reshape(-1).detach().cpu().numpy(), e_indices.reshape(-1)

        return z_e.reshape(-1).detach().cpu().numpy(), None

    def normalize_image(self, img):
        """normalizes image to [-1,1] interval

        Arguments:
            img {np.array or torch.tensor} -- [an image array / tensor with integer values 0-255]

        Returns:
            [np.array or torch tensor] -- [an image array / tensor with float values in [-1,1] interval]
        """
        # takes to [0,1] interval
        img /= 255.0
        # takes to [-0.5,0.5] interval
        img -= 0.5
        # takes to [-1,1] interval
        img /= 0.5
        return img

    def normalize_indices(self, x):
        assert max(x) <= 7.0, 'index mismatch during normalization'
        x = x.float()
        x /= 7.0
        x -= 0.5
        x /= 0.5
        return x

    def _get_pointmass_and_target_pos(self):
        target_pos = self.dm_env.physics.named.data.geom_xpos['target']
        pm_pos = self.dm_env.physics.named.data.geom_xpos['pointmass']
        return pm_pos, target_pos

    def reset(self):
        # generate goal, encode it, get continuous and discrete values
        self.encoded_goal, self.goal_indices = self.encode_observation(
            is_goal=True, include_indices=True)
        # reset env
        self.dm_env.reset()
        # get observation encoding
        z_e, e_indices = self.encode_observation(
            is_goal=False, include_indices=True)

        self.update_internal_state(e_indices, reset=True)

        obs_dict = self.construct_obs_dict(z_e, z_e, self.encoded_goal)

        return obs_dict

    def update_internal_state(self, e_indices, reset=False):
        if reset:
            self.steps = 0
            self.visited_reps = set()
            rep_hash = hash(tuple(e_indices))
            self.visited_reps.add(rep_hash)
            self.current_rep = rep_hash
            self.last_rep = 0
        else:
            self.steps += 1
            rep_hash = hash(tuple(e_indices))
            self.visited_reps.add(rep_hash)
            self.last_rep = self.current_rep
            self.current_rep = rep_hash

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


def raise_error(err_type='reward'):
    if err_type == 'reward':
        raise NotImplementedError('Must use dense or sparse reward')
    elif err_type == 'obs_dict':
        raise ValueError('obs must be a dict')
