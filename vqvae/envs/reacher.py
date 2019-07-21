from .utils import GoalBaseEnv, BaseEnv, L2_norm, GoalVQVAEEnv
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


class GoalReacher(GoalBaseEnv):
    def __init__(self,
                 max_steps=1000,
                 threshold=0.05,
                 reward_type='sparse',
                 img_dim=32,
                 camera_id=0
                 ):

        super().__init__(env_name='reacher', mode='easy', goal_dim=3)
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
        flat_obs = np.array(list(ts.observation.values())).reshape(-1)
        achieved_goal = self.dm_env.physics.named.data.geom_xpos['finger']
        obs_dict = self.construct_obs_dict(
            flat_obs, achieved_goal, self.desired_goal)
        return obs_dict

    def reset(self):
        ts, _, self.goal_img = self.reset_end_and_set_goal_image()
        self.desired_goal = self.dm_env.physics.named.data.geom_xpos['target'].copy(
        )
        # create desired image for goal

        flat_obs = np.array(list(ts.observation.values())).reshape(-1)
        achieved_goal = self.dm_env.physics.named.data.geom_xpos['finger'].copy(
        )
        obs_dict = self.construct_obs_dict(
            flat_obs, achieved_goal, self.desired_goal)

        self.update_internal_state(reset=True)
        return obs_dict

    def reset_end_and_set_goal_image(self):

        def l2(x, y):
            return np.linalg.norm(x-y)

        def obj_and_target_pos():
            xpos = self.dm_env.physics.named.data.geom_xpos
            return xpos['finger'], xpos['target']

        def move_reacher():
            with self.dm_env.physics.reset_context():
                self.dm_env.physics.named.data.qpos['shoulder'][0] = np.float(
                    np.random.randint(14))/2.0
                self.dm_env.physics.named.data.qpos['wrist'][0] = np.float(
                    np.random.randint(14))/2.0

        distance = float("inf")
        while distance > 0.05:
            ts = self.dm_env.reset()

            for _ in range(50):
                move_reacher()
                x, y = obj_and_target_pos()
                distance = l2(x, y)
                if distance <= 0.05:
                    break
        img = self.dm_env.physics.render(
            self.img_dim, self.img_dim, camera_id=self.camera_id)
        move_reacher()
        start_img = self.dm_env.physics.render(
            self.img_dim, self.img_dim, camera_id=self.camera_id)
        return ts, start_img, img

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


class GoalReacherNoTarget(GoalBaseEnv):
    def __init__(self,
                 max_steps=1000,
                 threshold=0.05,
                 reward_type='sparse',
                 img_dim=32,
                 camera_id=0
                 ):

        super().__init__(env_name='reacher', mode='no_target', goal_dim=3)
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
        flat_obs = np.array(list(ts.observation.values())).reshape(-1)
        achieved_goal = self.dm_env.physics.named.data.geom_xpos['finger']
        obs_dict = self.construct_obs_dict(
            flat_obs, achieved_goal, self.desired_goal)
        return obs_dict

    def reset(self):
        self.goal_img = self.reset_goal_image()
        self.desired_goal = self.dm_env.physics.named.data.geom_xpos['finger'].copy(
        )
        # create desired image for goal
        ts = self.dm_env.reset()

        flat_obs = np.array(list(ts.observation.values())).reshape(-1)
        achieved_goal = self.dm_env.physics.named.data.geom_xpos['finger'].copy(
        )
        obs_dict = self.construct_obs_dict(
            flat_obs, achieved_goal, self.desired_goal)

        self.update_internal_state(reset=True)
        return obs_dict

    def reset_goal_image(self):

        self.dm_env.reset()
        img = self.dm_env.physics.render(
            self.img_dim, self.img_dim, camera_id=self.camera_id)

        return img

    def update_internal_state(self, reset=False):
        if reset:
            self.steps = 0
        else:
            self.steps += 1

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


class GoalReacherVQVAE(GoalVQVAEEnv):
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
                 explore=False,
                 **kwargs):
        # be sure to specify obs dim and goal dim

        super().__init__(env_name='reacher', mode='easy',
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
        self.explore = explore
        self.rep_counts = {}

    def compute_reward(self, action, obs, *args, **kwargs):
        # abstract method only cares about obs and threshold

        distance = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])

        if self.reward_type == 'sparse':
            r = -1.0 if distance > self.threshold else 0.0
        elif self.reward_type == 'dense':
            r = - distance
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
        elif self.reward_type == 'continuous':
            r = -distances.astype(float)
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

    def step(self, a):
        obs_dict = self.step_and_get_obs_dict(a)
        reward = self.compute_reward(a, obs_dict)
        done, is_success = self.is_done(obs_dict)
        info = {
            'is_success': is_success,
            'num_reps': len(self.visited_reps),
            'latent_distance': np.linalg.norm(obs_dict['achieved_goal'] - obs_dict['desired_goal']),
            'rep_counts': len(self.rep_counts.keys())
        }

        return obs_dict, reward, done, info

    def step_and_get_obs_dict(self, action):
        assert self.encoded_goal is not None, "Must set desired goal before stepping"

        # 1. step forward with action
        # 2. get image, encode it

        _ = self.dm_env.step(action)

        z_e, e_indices = self.encode_observation(
            is_goal=False, include_indices=True)

        rep_hash = hash(tuple(e_indices))

        self.update_internal_state(rep_hash)

        obs_dict = self.construct_obs_dict(
            z_e, z_e, self.encoded_goal, rep_hash, self.goal_rep_hash)

        return obs_dict

    def encode_observation(self, is_goal=False, include_indices=True):
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

    def reset(self):
        # generate goal, encode it, get continuous and discrete values
        self.encoded_goal, self.goal_indices = self.encode_observation(
            is_goal=True, include_indices=True)
        self.goal_rep_hash = hash(tuple(self.goal_indices))
        # don't reset env again, since that will generate target in different position
        # get observation encoding
        z_e, e_indices = self.encode_observation(
            is_goal=False, include_indices=True)

        rep_hash = hash(tuple(e_indices))

        self.update_internal_state(rep_hash, reset=True)

        obs_dict = self.construct_obs_dict(
            z_e, z_e, self.encoded_goal, rep_hash, self.goal_rep_hash)

        return obs_dict

    def reset_goal_image(self):
        """
        Resets the environment and generates goal images, a bit tricky
        1. reset environment
        2. to generate goal image, move the arm randomly until it gets the target
        3. take an image snapshot of that state and set self.goal_img
        4. reset the arm (but not the target!) by assigning random values to its qpos joints


        Returns:
            np.array (float) -- goal image with shape (1,dim,dim,3)
        """
        def l2(x, y):
            return np.linalg.norm(x-y)

        def obj_and_target_pos():
            xpos = self.dm_env.physics.named.data.geom_xpos
            return xpos['finger'], xpos['target']

        def move_reacher():
            with self.dm_env.physics.reset_context():
                self.dm_env.physics.named.data.qpos['shoulder'][0] = np.float(
                    np.random.randint(14))/2.0
                self.dm_env.physics.named.data.qpos['wrist'][0] = np.float(
                    np.random.randint(14))/2.0

        distance = float("inf")
        while distance > 0.05:
            _ = self.dm_env.reset()

            for _ in range(50):
                move_reacher()
                x, y = obj_and_target_pos()
                distance = l2(x, y)
                if distance <= 0.05:
                    break
        goal_img = self.dm_env.physics.render(
            self.img_dim, self.img_dim, camera_id=self.camera_id)
        goal_img = goal_img.reshape(1, *goal_img.shape)
        self.goal_img = goal_img.copy()
        move_reacher()
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

    def _get_pointmass_and_target_pos(self):
        target_pos = self.dm_env.physics.named.data.geom_xpos['target']
        pm_pos = self.dm_env.physics.named.data.geom_xpos['pointmass']
        return pm_pos, target_pos

    def update_internal_state(self, rep_hash, reset=False):
        if reset:
            self.steps = 0
            self.visited_reps = set()
            self.visited_reps.add(rep_hash)
            self.current_rep = rep_hash
            self.last_rep = 0
        else:
            self.steps += 1
            self.visited_reps.add(rep_hash)
            self.last_rep = self.current_rep
            self.current_rep = rep_hash
        # increment count in rep_counts
        if rep_hash in self.rep_counts:
            self.rep_counts[rep_hash] += 1
        else:
            self.rep_counts[rep_hash] = 1

    def construct_obs_dict(self, obs, achieved_goal, desired_goal, achieved_rep_hash, desired_rep_hash):
        obs_dict = dict(
            observation=obs,
            state_observation=obs,
            achieved_goal=achieved_goal,
            state_achieved_goal=achieved_rep_hash,
            desired_goal=desired_goal,
            state_desired_goal=desired_rep_hash
        )
        return obs_dict


class GoalReacherNoTargetVQVAE(GoalVQVAEEnv):
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
                 explore=False,
                 **kwargs):
        # be sure to specify obs dim and goal dim

        super().__init__(env_name='reacher', mode='no_target',
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
        self.explore = explore
        self.rep_counts = {}

    def compute_reward(self, action, obs, *args, **kwargs):
        # abstract method only cares about obs and threshold

        distance = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'])

        if self.reward_type == 'sparse':
            r = -1.0 if distance > self.threshold else 0.0
        elif self.reward_type == 'dense':
            r = - distance
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
        elif self.reward_type == 'continuous':
            r = -distances.astype(float)
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

    def step(self, a):
        obs_dict = self.step_and_get_obs_dict(a)
        reward = self.compute_reward(a, obs_dict)
        done, is_success = self.is_done(obs_dict)
        info = {
            'is_success': is_success,
            'num_reps': len(self.visited_reps),
            'latent_distance': np.linalg.norm(obs_dict['achieved_goal'] - obs_dict['desired_goal']),
            'rep_counts': len(self.rep_counts.keys())
        }

        return obs_dict, reward, done, info

    def step_and_get_obs_dict(self, action):
        assert self.encoded_goal is not None, "Must set desired goal before stepping"

        # 1. step forward with action
        # 2. get image, encode it

        _ = self.dm_env.step(action)

        z_e, e_indices = self.encode_observation(
            is_goal=False, include_indices=True)

        rep_hash = hash(tuple(e_indices))

        self.update_internal_state(rep_hash)

        obs_dict = self.construct_obs_dict(
            z_e, z_e, self.encoded_goal, rep_hash, self.goal_rep_hash)

        return obs_dict

    def encode_observation(self, is_goal=False, include_indices=True):
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

    def reset(self):
        # generate goal, encode it, get continuous and discrete values
        self.encoded_goal, self.goal_indices = self.encode_observation(
            is_goal=True, include_indices=True)
        self.goal_rep_hash = hash(tuple(self.goal_indices))
        # don't reset env again, since that will generate target in different position
        # get observation encoding
        z_e, e_indices = self.encode_observation(
            is_goal=False, include_indices=True)

        rep_hash = hash(tuple(e_indices))

        self.update_internal_state(rep_hash, reset=True)

        obs_dict = self.construct_obs_dict(
            z_e, z_e, self.encoded_goal, rep_hash, self.goal_rep_hash)

        return obs_dict

    def reset_goal_image(self):
        """
        Resets the environment and generates goal images
        """
        # reset to get goal image
        self.dm_env.reset()
        goal_img = self.dm_env.physics.render(
            self.img_dim, self.img_dim, camera_id=self.camera_id)
        goal_img = goal_img.reshape(1, *goal_img.shape)
        self.goal_img = goal_img.copy()

        # reset to get different starting image
        self.dm_env.reset()
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

    def update_internal_state(self, rep_hash, reset=False):
        if reset:
            self.steps = 0
            self.visited_reps = set()
            self.visited_reps.add(rep_hash)
            self.current_rep = rep_hash
            self.last_rep = 0
        else:
            self.steps += 1
            self.visited_reps.add(rep_hash)
            self.last_rep = self.current_rep
            self.current_rep = rep_hash
        # increment count in rep_counts
        if rep_hash in self.rep_counts:
            self.rep_counts[rep_hash] += 1
        else:
            self.rep_counts[rep_hash] = 1

    def construct_obs_dict(self, obs, achieved_goal, desired_goal, achieved_rep_hash, desired_rep_hash):
        obs_dict = dict(
            observation=obs,
            state_observation=obs,
            achieved_goal=achieved_goal,
            state_achieved_goal=achieved_rep_hash,
            desired_goal=desired_goal,
            state_desired_goal=desired_rep_hash
        )
        return obs_dict


def raise_error(err_type='reward'):
    if err_type == 'reward':
        raise NotImplementedError('Must use dense or sparse reward')
    elif err_type == 'obs_dict':
        raise ValueError('obs must be a dict')
