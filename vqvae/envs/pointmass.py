from .utils import GoalBaseEnv, BaseEnv, L2_norm, VQVAEEnv
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


class EasyPointmassVQVAE(VQVAEEnv):
    def __init__(self,
                 max_steps=1000,
                 threshold=5e-2,
                 model_path='/home/misha/research/vqvae/results/vqvae_temporal_data_long_ne8nd2.pth',
                 **kwargs

                 ):
        self.max_steps = max_steps
        self.threshold = threshold
        super().__init__('point_mass', 'easy_big', model_path=model_path, **kwargs)
        self.steps = 0

    def reset_goal_image(self):

        super().reset()
        _, desired_goal = self._get_pointmass_and_target_pos()
        # some small amount of noise to generate realistic goal
        # scenarios
        noise = np.random.randn(3)*0.0015
        noise[-1] = 0.0
        # sets goal state
        self.dm_env.physics.named.data.geom_xpos['pointmass'] = desired_goal + noise
        goal_img = self.render(
            width=self.img_dim, height=self.img_dim, camera_id=self.camera_id)
        self.goal_img = goal_img
        goal_img = goal_img.reshape(1, *goal_img.shape)

        return goal_img

    def normalize_image(self, img):
        """normalizes image to [0,1] interval 

        Arguments:
            img {np.array or torch.tensor} -- [an image array / tensor with integer values 0-255]

        Returns:
            [np.array or torch tensor] -- [an image array / tensor with float values in [0,1] interval]
        """
        img /= 255.0
        img += 0.5
        img /= 0.5
        return img

    def reset(self):
        # generate goal, encode it, get continuous and discrete values
        self.encoded_goal, self.goal_indices = self.encode_observation(
            is_goal=True)
        # reset env
        super().reset()
        # get observation encoding
        z_e, e_indices = self.encode_observation(is_goal=False)

        return z_e.reshape(-1)

    def encode_observation(self, is_goal=False, include_indices=False):
        if is_goal:
            img = self.reset_goal_image()
            self.goal_img = goal_img
        else:
            img = self.get_current_image()

        img = self.numpy_to_tensor_img(img)
        img = self.normalize_image(img)

        z_e = self.encode_image(img)

        if include_indices:
            _, _, _, _, e_indices = model.vector_quantization(z_e)
            return z_e, e_indices

        return z_e, None

    def _get_pointmass_and_target_pos(self):
        target_pos = self.dm_env.physics.named.data.geom_xpos['target']
        pm_pos = self.dm_env.physics.named.data.geom_xpos['pointmass']
        return pm_pos, target_pos

    def compute_reward(self, action, z):
        return -np.log(100.0*self._distance_to_target(z)**2)

    def step(self, action):
        self.steps += 1
        _, r, d, info = super().step(action)
        z, e_indices = self.encode_observation(is_goal=False)

        r = self.compute_reward(action, z)
        info['is_success'] = self.is_success(z)
        d = self.is_done(z)
        if d:
            self.steps = 0
        return z.reshape(-1), r, d, info

    def is_success(self, z):
        return self._distance_to_target(z) < self.threshold

    def is_done(self, z):
        return self.is_success(z) or self.steps >= self.max_steps

    def _distance_to_target(self, z):
        x = self.encoded_goal
        return np.linalg.norm(z - x)


class Pointmass(GoalBaseEnv):
    def __init__(self,
                 max_steps=1000,
                 threshold=5e-2,
                 reward_type='dense',
                 fixed_length=False):

        super().__init__('point_mass', 'easy_big', 'pointmass', 'target', goal_dim=3)
        self.threshold = threshold
        self.max_steps = max_steps
        self.steps = 0
        self.fixed_length = fixed_length
        self.reward_type = reward_type

    def compute_reward(self, action, obs, *args, **kwargs):
        distance = L2_norm(obs['achieved_goal'], obs['desired_goal'])
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

    def step(self, action):
        self.steps += 1
        obs, _, _, info = super().step(action)
        if not isinstance(obs, dict):
            print(obs)
            raise_error(err_type='obs_dict')
        reward = self.compute_reward(action, obs)
        done, is_success = self._is_done(reward)

        info = {
            "is_success": is_success,
        }
        if done:
            self.steps = 0
        return obs, reward, done, info

    def _is_done(self, reward):
        # check if max step limit is reached
        done = False
        is_success = False
        if self.steps >= self.max_steps:
            done = True
            return done, is_success

        # check if episode was successful
        distance = self._distance_to_target()
        if self.reward_type == 'dense':
            is_success = distance < self.threshold
        elif self.reward_type == 'sparse':
            is_success = reward == 0
        else:
            raise_error(err_type='reward')
        done = is_success

        return done, is_success

    def reset(self):
        obs = super().reset()
        if not isinstance(obs, dict):
            print(obs)
            raise_error(err_type='obs_dict')
        return obs

    def _distance_to_target(self):
        x = self.dm_env.physics.named.data.geom_xpos
        return np.linalg.norm(x['target'] - x['pointmass'])


def raise_error(err_type='reward'):
    if err_type == 'reward':
        raise NotImplementedError('Must use dense or sparse reward')
    elif err_type == 'obs_dict':
        raise ValueError('obs must be a dict')
