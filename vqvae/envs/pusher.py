from .utils import GoalBaseEnv, BaseEnv, L2_norm, GoalVQVAEEnv
import numpy as np
from .utils import _ts_to_obs
import time


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


class GoalPusherNoTarget(GoalBaseEnv):
    def __init__(self,
                 max_steps=1000,
                 threshold=0.05,
                 reward_type='sparse',
                 img_dim=32,
                 camera_id=0,
                 obs_dim=42,
                 goal_dim=3
                 ):

        super().__init__(env_name='stacker', mode='push_1',
                         goal_dim=goal_dim, obs_dim=obs_dim)
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
        flat_obs = np.append(
            flat_obs, [self.desired_goal[0], self.desired_goal[2]])
        achieved_goal = self.get_achieved_goal()
        obs_dict = self.construct_obs_dict(
            flat_obs, achieved_goal, self.desired_goal)
        return obs_dict

    def reset(self):
        obs_dict, desired_goal = stationary_easy_reset(self.dm_env)
        self.desired_goal = desired_goal
        return obs_dict

    def old_reset(self):

        flat_obs = stationary_reset(self.dm_env)
        self.desired_goal = self.get_desired_goal(reset=True)
        flat_obs = np.append(
            flat_obs, [self.desired_goal[0], self.desired_goal[2]])

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

    def get_desired_goal(self, reset=False):
        if reset:
            box_size = self.dm_env.physics.named.model.geom_size['box0', 0]
            desired_goal = np.array(
                [np.array(np.random.uniform(-.37, .37)), 0.001, box_size])
        else:
            desired_goal = self.desired_goal
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


class GoalPusherNoTargetVQVAE(GoalVQVAEEnv):
    def __init__(self,
                 obs_dim=None,
                 goal_dim=None,
                 model_path=None,
                 img_dim=64,
                 camera_id=0,
                 gpu_id=0,
                 reward_type='real_sparse',
                 rep_type='continuous',
                 max_steps=500,
                 threshold=0.05,
                 e_index_dim=8.0,
                 explore=False,
                 **kwargs):
        # be sure to specify obs dim and goal dim

        super().__init__(env_name='stacker', mode='push_1',
                         obs_dim=obs_dim, act_dim=None, goal_dim=goal_dim,
                         model_path=model_path,
                         img_dim=img_dim,
                         camera_id=camera_id,
                         gpu_id=gpu_id)

        self.threshold = threshold
        self.e_index_dim = e_index_dim
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

        if 'real' in self.reward_type:
            real_distance = np.linalg.norm(
                obs['state_achieved_goal'] - obs['state_desired_goal'])
            if self.reward_type == 'real_sparse':
                r = -1.0 if real_distance > self.threshold else 0.0
            elif self.reward_type == 'real_dense':
                r = - real_distance
        elif self.reward_type == 'latent_sparse':
            latent_distance = np.linalg.norm(
                obs['achieved_goal'] - obs['desired_goal'])
            r = -1.0 if latent_distance > self.threshold else 0.0
        else:
            raise_error(err_type='reward')

        return r

    def compute_rewards(self, actions, obs):
        # abstract method only cares about obs and threshold

        if 'real' in self.reward_type:
            real_distances = np.linalg.norm(
                obs['state_achieved_goal'] - obs['state_desired_goal'], axis=1)
            if self.reward_type == 'real_sparse':
                r = -(real_distances > self.threshold).astype(float)
            elif self.reward_type == 'real_dense':
                r = - real_distances.astype(float)
        elif self.reward_type == 'latent_sparse':
            latent_distances = np.linalg.norm(
                obs['achieved_goal'] - obs['desired_goal'], axis=1)
            r = -(latent_distances > self.threshold).astype(float)
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
        if 'real' in self.reward_type:
            real_distance = np.linalg.norm(
                obs['state_achieved_goal'] - obs['state_desired_goal'])
            is_success = real_distance < self.threshold
        elif 'latent' in self.reward_type:
            latent_distance = np.linalg.norm(
                obs['achieved_goal'] - obs['desired_goal'])
            is_success = latent_distance < self.threshold

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
            'real_distance': np.linalg.norm(obs_dict['state_achieved_goal'] - obs_dict['state_desired_goal']),
            'rep_counts': len(self.rep_counts.keys())
        }

        return obs_dict, reward, done, info

    def step_and_get_obs_dict(self, action):
        assert self.encoded_goal is not None, "Must set desired goal before stepping"

        # 1. step forward with action
        # 2. get image, encode it

        ts = self.dm_env.step(action)
        flat_obs = _ts_to_obs(ts)
        flat_obs = np.append(
            flat_obs, [self.state_desired_goal[0], self.state_desired_goal[2]])
        z_e, e_indices = self.encode_observation(include_indices=True)

        rep_hash = hash(tuple(e_indices))

        self.update_internal_state(rep_hash)

        box_pos = self.dm_env.physics.named.data.geom_xpos["box0"].copy()

        obs_dict = dict(
            observation=z_e,
            achieved_goal=z_e,
            desired_goal=self.encoded_goal,
            state_observation=flat_obs,
            state_achieved_goal=box_pos,
            state_desired_goal=self.state_desired_goal
        )

        return obs_dict

    def encode_observation(self, goal_img=None, include_indices=True, normalize_e=False):
        if goal_img is not None:
            img = goal_img
        else:
            img = self.get_current_image()

        img = self.numpy_to_tensor_img(img)
        img = self.normalize_image(img)

        z_e = self.encode_image(img, as_tensor=True)

        if include_indices:
            _, _, _, _, e_indices = self.model.vector_quantization(z_e)
            if normalize_e:
                e_indices = self.normalize_indices(
                    e_indices).detach().cpu().numpy()
            else:
                e_indices = e_indices.detach().cpu().numpy()
            return z_e.reshape(-1).detach().cpu().numpy(), e_indices.reshape(-1)

        return z_e.reshape(-1).detach().cpu().numpy(), None

    def reset(self):
        # generate goal, encode it, get continuous and discrete values
        self.goal_img, flat_obs = self.reset_goal_image()
        self.encoded_goal, self.goal_indices = self.encode_observation(
            goal_img=self.goal_img, include_indices=True)
        self.goal_rep_hash = hash(tuple(self.goal_indices))
        # don't reset env again, since that will generate target in different position
        # get observation encoding
        z_e, e_indices = self.encode_observation(include_indices=True)

        rep_hash = hash(tuple(e_indices))

        self.update_internal_state(rep_hash, reset=True)

        box_pos = self.dm_env.physics.named.data.geom_xpos["box0"].copy()

        obs_dict = dict(
            observation=z_e,
            achieved_goal=z_e,
            desired_goal=self.encoded_goal,
            state_observation=flat_obs,
            state_achieved_goal=box_pos,
            state_desired_goal=self.state_desired_goal
        )

        return obs_dict

    def reset_goal_image(self):
        """
        Resets the environment and generates goal images

        we'll want to 
        1. do a stationary reset to get the block flat
        2. sample block x position uniformly
        3. set gripper to default configuration (left or right)
        4. take image snapshot for goal 
        5. reset the thing and start block on opposite side of desired location
        """
        # reset to get goal image
        # step 1
        flat_obs = stationary_reset(self.dm_env)
        original_box_pos = get_xpos(self.dm_env, name='box0').copy()
        # step 2
        index = np.random.randint(2)
        ranges = [[-.37, -.3], [.3, .37]]
        range_ = ranges[index]
        box_pos = np.random.uniform(*range_)
        set_pusher_qpos(self.dm_env, box0_x=box_pos)
        self.state_desired_goal = get_xpos(self.dm_env, name='box0').copy()

        """
        set_arm_on_right = dict(
            arm_root=-1.6, arm_shoulder=-.5, arm_elbow=-1, arm_wrist=-1)
        set_arm_on_left = dict(
            arm_root=1.6, arm_shoulder=.5, arm_elbow=1, arm_wrist=1)

        if box_pos < 0:
            set_pusher_qpos(self.dm_env, **set_arm_on_right)
        else:
            set_pusher_qpos(self.dm_env, **set_arm_on_left)
        """
        # step 4
        goal_img = self.dm_env.physics.render(
            self.img_dim, self.img_dim, camera_id=self.camera_id)
        goal_img = goal_img.reshape(1, *goal_img.shape)

        # reset to get different starting image
        #left_curriculum = [[-.35,.25],[-.35,-.15],[-.35,-.05],[-.35,0.05],[-.35,]]
        #right_curriculum = [[]]

        # return to original state
        set_pusher_qpos(self.dm_env, box0_x=original_box_pos[0])

        flat_obs = np.append(
            flat_obs, [self.state_desired_goal[0], self.state_desired_goal[2]])
        """
        stationary_reset(self.dm_env)
        starting_box_pos = np.random.uniform(-.37, .37)
        set_pusher_qpos(self.dm_env, box0_x=starting_box_pos)
        """
        return goal_img, flat_obs

    def old_reset_goal_image(self):
        """
        Resets the environment and generates goal images

        we'll want to 
        1. do a stationary reset to get the block flat
        2. sample block x position uniformly
        3. set gripper to default configuration (left or right)
        4. take image snapshot for goal 
        5. reset the thing and start block on opposite side of desired location
        """
        # reset to get goal image
        # step 1
        stationary_reset(self.dm_env)
        # step 2
        box_pos = np.random.uniform(-.37, .37)
        self.state_desired_goal = self.dm_env.physics.named.data.geom_xpos["box0"].copy(
        )
        set_pusher_qpos(self.dm_env, box0_x=box_pos)
        # step 3
        set_arm_on_right = dict(
            arm_root=-1.6, arm_shoulder=-.5, arm_elbow=-1, arm_wrist=-1)
        set_arm_on_left = dict(
            arm_root=1.6, arm_shoulder=.5, arm_elbow=1, arm_wrist=1)

        if box_pos < 0:
            set_pusher_qpos(self.dm_env, **set_arm_on_right)
        else:
            set_pusher_qpos(self.dm_env, **set_arm_on_left)

        # step 4
        goal_img = self.dm_env.physics.render(
            self.img_dim, self.img_dim, camera_id=self.camera_id)
        goal_img = goal_img.reshape(1, *goal_img.shape)

        # reset to get different starting image
        stationary_reset(self.dm_env)
        if box_pos < 0:
            starting_box_pos = np.random.uniform(0.0, .37)
        else:
            starting_box_pos = np.random.uniform(-.37, 0.0)
        set_pusher_qpos(self.dm_env, box0_x=starting_box_pos)
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
        assert max(x) < self.e_index_dim, 'index mismatch during normalization'
        x = x.float()
        x /= self.e_index_dim-1
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





"""
Utilities
"""


def rand_a(spec):
    return np.random.uniform(spec.minimum, spec.maximum, spec.shape)


def dict_to_flat_arr(d):
    flat_arrs = np.concatenate([np.array(v).reshape(-1)
                                for v in list(d.values())]).reshape(-1)
    return np.array([v for v in flat_arrs])


def stationary_easy_reset(env,n_steps=10):
    # first put in easy position
    geom_xpos = env.physics.named.data.geom_xpos
    physics = env.physics
    data = env.physics.named.data
    
    start = time.time()
    
    while True:
        ts = env.reset()
        spec = env.action_spec()
        f1 = geom_xpos['fingertip1'][-1]
        f2 = geom_xpos['fingertip2'][-1]
        t1= geom_xpos['thumbtip1'][-1]
        t2= geom_xpos['thumbtip2'][-1]
        fi1 = geom_xpos['finger1'][-1]
        fi2 = geom_xpos['finger2'][-1]
        ti1= geom_xpos['thumb1'][-1]
        ti2= geom_xpos['thumb2'][-1]
        parts = dict(fingertip1=f1,fingertip2=f2,thumbtip1=t1,thumbtip2=t2,thumb1=ti1,thumb2=ti2,finger1=fi1,finger2=fi2)
        mean_part = np.mean(list(parts.values())).copy()
        
        epsilons = [0.2 for _ in range(len(parts.values()))]
        if sum(np.less(list(parts.values()),epsilons)):
            obj = list(parts.keys())[np.random.randint(len(parts.keys()))]
            obj_pos = .5*(geom_xpos['thumbtip2'][0]+geom_xpos['fingertip2'][0]) #geom_xpos[obj][0]
            x = obj_pos #+ np.random.uniform(.05,.1)*(np.random.randint(2)-1)
            data.qpos['box0_x'] = x
            data.qpos['box0_z'] = np.random.uniform(0.033,.05)
            physics.after_reset()
            penetrating = physics.data.ncon > 0
            break
            if not penetrating:
                for _ in range(20):
                    a = np.zeros(spec.shape)
                    ts = env.step(a)
                break
                
    
    f1_x = geom_xpos['fingertip1'][0].copy()
    box0_x = geom_xpos['box0'][0].copy()
    
    hand_on_right = f1_x > box0_x
    
    if hand_on_right:
        x_goal = np.random.uniform(-.36,box0_x)
    else:
        x_goal = np.random.uniform(box0_x,.36)
    
    box_goal = np.array([x_goal,0.001,0.033])
    box_pos = geom_xpos["box0"].copy()
    ts_dict = ts.observation.copy()
    ts_dict["box_pos"][0][:2] = [box_pos[0],box_pos[2]]
    ts_dict['target'] = [box_goal[0],box_goal[2]]
    flat_obs = dict_to_flat_arr(ts_dict)
    
    obs_dict = dict(
        observation=flat_obs,
        achieved_goal=box_pos,
        desired_goal = box_goal,
        state_observation=flat_obs,
        state_achieved_goal=box_pos,
        state_desired_goal = box_goal,
    )
    return obs_dict, box_goal


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


def set_qpos(env, **kwargs):
    qpos = env.physics.data.qpos.copy()
    model = env.physics.model
    nq = model.nq
    with env.physics.reset_context():
        for i in range(nq):
            env.physics.data.qpos[i] = kwargs[str(i)] if str(
                i) in kwargs else qpos[i]


def set_pusher_qpos(env, **kwargs):
    id2name_list = ['arm_root', 'arm_shoulder', 'arm_elbow', 'arm_wrist', 'thumb',
                    'thumbtip', 'finger', 'fingertip', 'box0_x', 'box0_y', 'box0_z']
    id2val = {str(id2name_list.index(k)): v for k, v in kwargs.items()}

    set_qpos(env, **id2val)


def get_xpos(env, name=None):
    return env.physics.named.data.geom_xpos[name]


def raise_error(err_type='reward'):
    if err_type == 'reward':
        raise NotImplementedError('Must use dense or sparse reward')
    elif err_type == 'obs_dict':
        raise ValueError('obs must be a dict')
