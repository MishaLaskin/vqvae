from dm_control import suite
from gym.spaces import Box, Dict
import numpy as np
from functools import reduce
from vqvae.models.vqvae import VQVAE, TemporalVQVAE
from vqvae import utils
import torch
# BASE class


class BaseEnv:
    """
    Base class to be used as parent for DM control envs
    Built to be compatible with Gym API and Rlkit

    Returns:
        [class] -- [env class with reset, step, and render methods]
    """

    def __init__(self,
                 env_name, mode,
                 obs_dim=None, act_dim=None,
                 **kwargs):

        # loads env from dm_control
        self.dm_env = suite.load(env_name, mode)
        self.task = self.dm_env._task

        # takes in all obs shapes and sums them
        obs_shapes = [v.shape for v in list(
            self.dm_env.observation_spec().values())]
        obs_shapes_sum = np.sum(
            [reduce((lambda x, y: x * y), z) for z in obs_shapes])

        # actions always between -1.0 and 1.0
        self.action_space = Box(
            high=1.0, low=-1.0,
            shape=self.dm_env.action_spec().shape if act_dim is None else (act_dim,)
        )

        # observtions are, in principle, unbounded
        self.observation_space = Box(
            high=float("inf"), low=-float("inf"),
            shape=((obs_shapes_sum,) if obs_dim is None else (obs_dim,)))

    def reset(self):
        # resets env by returning a flat obs (not a dict)
        ts = self.dm_env.reset()
        obs = _ts_to_obs(ts)
        return obs

    def step(self, action):
        # one timestep forward
        # reward and done are taken from dm_control's env
        ts = self.dm_env.step(action)
        obs = _ts_to_obs(ts)
        reward, done = ts.reward, ts.last()
        info = {}
        return obs, reward, done, info

    def physics(self):
        # convenience method for accessing physics
        return self.dm_env.physics

    def render(self, **render_kwargs):
        # renders image
        # example: render_kwargs={width=32,height=32,camera_id=0}
        return self.physics().render(**render_kwargs)


class GoalBaseEnv(BaseEnv):

    def __init__(self,
                 env_name=None, mode=None,
                 obs_dim=None, act_dim=None, goal_dim=None,
                 **kwargs):

        super().__init__(env_name, mode, obs_dim=obs_dim,
                         act_dim=act_dim, **kwargs)

        if goal_dim is None:
            goal_dim = self.observation_space.shape[0]

        goal_space = Box(
            high=float("inf"), low=-float("inf"),
            shape=(goal_dim,)
        )

        flat_obs_space = self.observation_space

        self.observation_space = Dict([
            ('observation', flat_obs_space),
            ('desired_goal', goal_space),
            ('achieved_goal', goal_space),
            ('state_observation', flat_obs_space),
            ('state_desired_goal', goal_space),
            ('state_achieved_goal', goal_space),
        ])

    def step(self, a):
        obs_dict = self.step_and_get_obs_dict(a)
        reward = self.compute_reward(a, obs_dict)
        done, is_success = self.is_done(obs_dict)
        info = {
            'is_success': is_success
        }
        self.update_internal_state()
        return obs_dict, reward, done, info

    def reset(self):
        raise NotImplementedError

    def update_internal_state(self):
        raise NotImplementedError

    def compute_reward(self, action, obs_dict, *args, **kwargs):
        raise NotImplementedError

    def step_and_get_obs_dict(self, a):
        raise NotImplementedError

    def is_done(self, obs_dict):
        raise NotImplementedError


class VQVAEEnv(BaseEnv):
    def __init__(self,
                 env_name, mode,
                 obs_dim=None, act_dim=None,
                 model_path=None,
                 img_dim=32,
                 camera_id=0,
                 gpu_id=0,
                 **kwargs):

        super().__init__(env_name, mode, obs_dim=obs_dim, act_dim=act_dim, **kwargs)
        assert model_path is not None, 'Must specify model path'
        self.device = torch.device("cuda:"+str(gpu_id)
                                   if torch.cuda.is_available() else "cpu")

        self.model, self.model_params = load_model(
            model_path, gpu_id, self.device)
        self.img_dim = img_dim
        self.camera_id = 0

    def reset(self):
        # set goal

        # reset env
        super().reset()

        img = self.get_current_image()
        img = self.numpy_to_tensor_img(img)
        z_e = self.encode_image(img)
        return z_e

    def encode_image(self, img, as_tensor=False):
        z_e = self.model.pre_quantization_conv(
            self.model.encoder(img))
        if as_tensor:
            return z_e
        return z_e.detach().cpu().numpy()

    def get_current_image(self):
        img = self.render(width=self.img_dim,
                          height=self.img_dim, camera_id=self.camera_id)
        img = img.reshape(1, *img.shape)
        return img

    def get_goal_image(self):
        raise NotImplementedError

    def numpy_to_tensor_img(self, img):
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float().to(self.device)
        return img.permute(0, 3, 1, 2)


class GoalVQVAEEnv(GoalBaseEnv):

    def __init__(self,
                 env_name=None, mode=None,
                 obs_dim=None, act_dim=None, goal_dim=None,
                 model_path=None,
                 img_dim=32,
                 camera_id=0,
                 gpu_id=0, **kwargs):
        super().__init__(env_name=env_name, mode=mode, obs_dim=obs_dim,
                         act_dim=act_dim, goal_dim=goal_dim, **kwargs)

        assert model_path is not None, 'Must specify model path'
        self.device = torch.device("cuda:"+str(gpu_id)
                                   if torch.cuda.is_available() else "cpu")

        self.model, self.model_params = load_model(
            model_path, gpu_id, self.device)
        self.img_dim = img_dim
        self.camera_id = 0

    def encode_image(self, img, as_tensor=False):
        z_e = self.model.pre_quantization_conv(
            self.model.encoder(img))
        if as_tensor:
            return z_e
        return z_e.detach().cpu().numpy()

    def get_current_image(self):
        img = self.render(width=self.img_dim,
                          height=self.img_dim, camera_id=self.camera_id)
        img = img.reshape(1, *img.shape)
        return img

    def get_goal_image(self):
        raise NotImplementedError

    def numpy_to_tensor_img(self, img):
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float().to(self.device)
        return img.permute(0, 3, 1, 2)


def _has_attr(obj, name):
    attr = getattr(obj, name, None)
    return True if callable(attr) else False


def _ts_to_obs(ts):
    flat_obs = np.concatenate([v.reshape(-1)
                               for v in list(ts.observation.values())]).reshape(-1)
    return np.array([v for v in flat_obs])


def _obs_to_ground_truth_dict(obs, physics, object, target):
    desired_goal = physics.named.data.geom_xpos[target]
    achieved_goal = physics.named.data.geom_xpos[object]

    goal_obs = {
        "observation": obs,
        "desired_goal": desired_goal,
        "achieved_goal": achieved_goal,
        "state_observation": obs,
        "state_desired_goal": desired_goal,
        "state_achieved_goal": achieved_goal,
    }
    return goal_obs


def L2_norm(x1, x2):
    return np.linalg.norm(x1-x2)


def load_model(model_path, gpu_id, device):
    """
    Loads the VQ VAE model
    if running from the VQ VAE folder, you can set model_dir = None
    """

    path = model_path

    if torch.cuda.is_available():
        data = torch.load(path)
    else:
        data = torch.load(path, map_location=lambda storage, loc: storage)

    params = data["hyperparameters"]

    model = VQVAE(params['n_hiddens'], params['n_residual_hiddens'],
                  params['n_residual_layers'], params['n_embeddings'],
                  params['embedding_dim'], params['beta'], gpu_id=gpu_id).to(device)

    model.load_state_dict(data['model'])

    return model, params


class SimpleGoalEnv:

    def __init__(self, obs_dim=42, goal_dim=3,
                 env_name='stacker', task='push_1',
                 max_steps=200, reward_type='dense', threshold=0.05):

        self.max_steps = max_steps
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.dm_env = suite.load(env_name, task)
        self.physics = self.dm_env.physics
        self.model = self.physics.named.model
        self.data = self.physics.named.data
        self.box_size = self.model.geom_size['box0', 0]
        self.reward_type = reward_type
        accepted_reward_types = ["dense", "sparse"]
        assert self.reward_type in accepted_reward_types, "Invalid reward type"
        self.threshold = threshold
        #
        self.act_dim = self.dm_env.action_spec().shape[0]
        # actions always between -1.0 and 1.0
        self.action_space = Box(
            high=1.0, low=-1.0,
            shape=(self.act_dim,)
        )

        # observtions are, in principle, unbounded
        flat_obs_space = Box(
            high=float("inf"), low=-float("inf"),
            shape=(obs_dim,))

        goal_space = Box(
            high=float("inf"), low=-float("inf"),
            shape=(goal_dim,))

        self.observation_space = Dict([
            ('observation', flat_obs_space),
            ('desired_goal', goal_space),
            ('achieved_goal', goal_space),
            ('state_observation', flat_obs_space),
            ('state_desired_goal', goal_space),
            ('state_achieved_goal', goal_space)
        ])

    def reset(self):
        ts = self.dm_env.reset()
        self.steps = 0
        hand_pos = self.data.geom_xpos["hand"].copy()
        box_pos = self.data.geom_xpos["box0"].copy()
        if hand_pos[0] > box_pos[0]:
            goal_x = np.random.uniform(-.37,
                                       np.clip(box_pos[0]-0.05, -.33, .33))
        else:
            goal_x = np.random.uniform(
                np.clip(box_pos[0]+0.05, -.33, .33), .37)
        self.goal = np.array([goal_x, 0.001, self.box_size])
        flat_obs = dict_to_flat_arr(ts.observation)
        flat_obs = np.append(flat_obs, [self.goal[0], self.goal[2]])

        obs_dict = dict(
            observation=flat_obs,
            desired_goal=self.goal,
            achieved_goal=box_pos,
            state_observation=flat_obs,
            state_desired_goal=self.goal,
            state_achieved_goal=box_pos
        )
        return obs_dict

    def compute_reward(self, action, obs_dict):
        distance = np.linalg.norm(
            obs_dict["desired_goal"] - obs_dict["achieved_goal"])
        if self.reward_type == 'dense':
            return -distance.astype(float)
        if self.reward_type == 'sparse':
            return - (distance > self.threshold).astype(float)

    def compute_rewards(self, actions, obs_dict):
        distances = np.linalg.norm(
            obs_dict["desired_goal"] - obs_dict["achieved_goal"], axis=1)
        if self.reward_type == 'dense':
            return -distances.astype(float)
        if self.reward_type == 'sparse':
            return - (distances > self.threshold).astype(float)

    def is_done(self, obs_dict):
        distance = np.linalg.norm(
            obs_dict["desired_goal"] - obs_dict["achieved_goal"])

        if distance < self.threshold:
            return True, True
        elif self.steps >= self.max_steps:
            return True, False
        else:
            return False, False

    def step(self, action):
        ts = self.dm_env.step(action)
        self.steps += 1
        flat_obs = dict_to_flat_arr(ts.observation)
        flat_obs = np.append(flat_obs, [self.goal[0], self.goal[2]])
        box_pos = self.data.geom_xpos["box0"].copy()
        obs_dict = dict(
            observation=flat_obs,
            desired_goal=self.goal,
            achieved_goal=box_pos,
            state_observation=flat_obs,
            state_desired_goal=self.goal,
            state_achieved_goal=box_pos
        )

        reward = self.compute_reward(action, obs_dict)
        done, is_success = self.is_done(obs_dict)
        info = dict(is_success=is_success)
        return obs_dict, reward, done, info

    def render(self, **render_kwargs):
        return self.dm_env.physics.render(**render_kwargs)


class LatentGoalEnv:

    def __init__(self, obs_dim=128, goal_dim=128,
                 env_name='stacker', task='push_1',
                 max_steps=100, reward_type='sparse', threshold=0.1,
                 render_kwargs=dict(width=64, height=64, camera_id=0),
                 gpu_id=0, easy_reset=False

                 ):

        self.max_steps = max_steps
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.dm_env = suite.load(env_name, task)
        self.physics = self.dm_env.physics
        self.model = self.physics.named.model
        self.data = self.physics.named.data
        self.box_size = self.model.geom_size['box0', 0]
        self.reward_type = reward_type
        self.render_kwargs = render_kwargs
        self.easy_reset = easy_reset
        accepted_reward_types = ["dense", "sparse"]
        assert self.reward_type in accepted_reward_types, "Invalid reward type"
        self.threshold = threshold

        self.device = torch.device("cuda:"+str(gpu_id)
                                   if torch.cuda.is_available() else "cpu")
        #
        self.act_dim = self.dm_env.action_spec().shape[0]
        # actions always between -1.0 and 1.0
        self.action_space = Box(
            high=1.0, low=-1.0,
            shape=(self.act_dim,)
        )

        # observtions are, in principle, unbounded
        flat_obs_space = Box(
            high=float("inf"), low=-float("inf"),
            shape=(obs_dim,))

        goal_space = Box(
            high=float("inf"), low=-float("inf"),
            shape=(goal_dim,))

        self.observation_space = Dict([
            ('observation', flat_obs_space),
            ('desired_goal', goal_space),
            ('achieved_goal', goal_space),
            ('state_observation', flat_obs_space),
            ('state_desired_goal', goal_space),
            ('state_achieved_goal', goal_space)
        ])

        model1_filename = '/home/misha/downloads/vqvae/results/vqvae_data_pusher_jul25_ne128nd2.pth'
        model2_filename = '/home/misha/downloads/vqvae/results/vqvae_data_block_jul25_ne128nd2.pth'

        self.model1, _ = utils.load_model(
            model1_filename, temporal=False)

        self.model2, _ = utils.load_model(
            model2_filename, temporal=False)

        self.block_env = RefBlockEnv()

    def reset(self):
        """ reset and get state obs, state ag, and state dg """
        ts = self.dm_env.reset()
        self.steps = 0
        hand_pos = self.data.geom_xpos["hand"].copy()
        box_pos = self.data.geom_xpos["box0"].copy()
        if hand_pos[0] > box_pos[0]:
            goal_x = np.random.uniform(-.37,
                                       np.clip(box_pos[0]-0.05, -.33, .33))
            if self.easy_reset:
                goal_x = np.clip(box_pos[0] - np.random.uniform(2*0.033,.1+0.033),-.33,.33)

        else:
            goal_x = np.random.uniform(
                np.clip(box_pos[0]+0.05, -.33, .33), .37)
            if self.easy_reset:
                goal_x = np.clip(box_pos[0] + np.random.uniform(2*0.033,.1+0.033),-.33,.33)
        self.goal = np.array([goal_x, 0.001, self.box_size])
        flat_obs = dict_to_flat_arr(ts.observation)
        flat_obs = np.append(flat_obs, [self.goal[0], self.goal[2]])

        """ get image and encode it to get obs """
        img = self.render(**self.render_kwargs)
        obs_z = self.img_to_latent(img, self.model1)

        """ use block env to set desired goal encoding and encode achieved goal """
        self.block_env.reset()
        # desired goal
        self.block_env.set_block_pos(x=self.goal[0], z=self.goal[2])
        goal_img = self.block_env.render(**self.render_kwargs)
        self.goal_image = goal_img

        goal_z = self.img_to_latent(goal_img, self.model2)
        self.latent_goal = goal_z

        # achieved goal
        self.block_env.set_block_pos(x=box_pos[0], z=box_pos[2])
        achieved_img = self.block_env.render(**self.render_kwargs)
        achieved_z = self.img_to_latent(achieved_img, self.model2)

        obs_dict = dict(
            observation=obs_z,
            desired_goal=self.latent_goal,
            achieved_goal=achieved_z,
            state_observation=flat_obs,
            state_desired_goal=self.goal,
            state_achieved_goal=box_pos
        )
        return obs_dict

    def img_to_latent(self, img, model):
        img = numpy_to_tensor_img(img)
        img = img.to(self.device)
        img = normalize_image(img)
        z = model.pre_quantization_conv(model.encoder(img))
        return z.detach().cpu().numpy().reshape(-1)

    def compute_reward(self, action, obs_dict):
        distance = np.linalg.norm(
            obs_dict["desired_goal"] - obs_dict["achieved_goal"])
        if self.reward_type == 'dense':
            return -distance.astype(float)
        if self.reward_type == 'sparse':
            return - (distance > self.threshold).astype(float)

    def compute_rewards(self, actions, obs_dict):
        distances = np.linalg.norm(
            obs_dict["desired_goal"] - obs_dict["achieved_goal"], axis=1)
        if self.reward_type == 'dense':
            return -distances.astype(float)
        if self.reward_type == 'sparse':
            return - (distances > self.threshold).astype(float)

    def is_done(self, obs_dict):
        distance = np.linalg.norm(
            obs_dict["desired_goal"] - obs_dict["achieved_goal"])

        if distance < self.threshold:
            return True, True
        elif self.steps >= self.max_steps:
            return True, False
        else:
            return False, False

    def step(self, action):
        ts = self.dm_env.step(action)
        self.steps += 1
        flat_obs = dict_to_flat_arr(ts.observation)
        flat_obs = np.append(flat_obs, [self.goal[0], self.goal[2]])
        box_pos = self.data.geom_xpos["box0"].copy()

        """ get image and encode it to get obs """
        img = self.render(**self.render_kwargs)
        obs_z = self.img_to_latent(img, self.model1)

        """ use block env to set block and get achieved goal """

        # achieved goal
        self.block_env.set_block_pos(x=box_pos[0], z=box_pos[2])
        achieved_img = self.block_env.render(**self.render_kwargs)
        achieved_z = self.img_to_latent(achieved_img, self.model2)

        obs_dict = dict(
            observation=obs_z,
            desired_goal=self.latent_goal,
            achieved_goal=achieved_z,
            state_observation=flat_obs,
            state_desired_goal=self.goal,
            state_achieved_goal=box_pos
        )

        reward = self.compute_reward(action, obs_dict)
        done, is_success = self.is_done(obs_dict)
        info = dict(is_success=is_success)
        return obs_dict, reward, done, info

    def render(self, **render_kwargs):
        return self.dm_env.physics.render(**render_kwargs)


class RefBlockEnv:

    def __init__(self, env_name='blocks', task='push_1', render_kwargs=dict(width=64, height=64, camera_id=0)):

        self.dm_env = suite.load(env_name, task)
        self.physics = self.dm_env.physics
        self.model = self.physics.named.model
        self.data = self.physics.named.data
        self.box_size = self.model.geom_size['box0', 0]
        self.render_kwargs = render_kwargs

    def set_block_pos(self, x=0.0, y=None, z=None):
        with self.physics.reset_context():
            self.data.qpos["box0_x"] = x
            self.data.qpos["box0_y"] = 0.001 if y is None else y
            self.data.qpos["box0_z"] = self.box_size if z is None else z

    def render(self, **render_kwargs):
        kwargs = render_kwargs if render_kwargs else self.render_kwargs
        return self.dm_env.physics.render(**kwargs)

    def reset(self):
        return self.dm_env.reset()

    def step(self, a):
        return self.dm_env.step(a)

class RefTwoBlocksEnv:

    def __init__(self, env_name='blocks', task='stack_2', render_kwargs=dict(width=64, height=64, camera_id=0)):

        self.dm_env = suite.load(env_name, task)
        self.physics = self.dm_env.physics
        self.model = self.physics.named.model
        self.data = self.physics.named.data
        self.box_size = self.model.geom_size['box0', 0]
        self.render_kwargs = render_kwargs
        self.red = np.array([246.,91.,75.])/255.
        self.green = np.array([93.,205.,189.])/255.

    def set_block_pos(self, box0=[None,None,None],box1=[None,None,None]):
        
        xyz0 = [self.data.qpos["box0_x"].copy(),
                self.data.qpos["box0_y"].copy(),
                self.data.qpos["box0_z"].copy()]

        xyz1 = [self.data.qpos["box1_x"].copy(),
                self.data.qpos["box1_y"].copy(),
                self.data.qpos["box1_z"].copy()]

        with self.physics.reset_context():
            self.data.qpos["box0_x"] = xyz0[0] if box0[0] is None else box0[0]
            self.data.qpos["box0_y"] = 0.001 if box0[1] is None else box0[1]
            self.data.qpos["box0_z"] = xyz0[2] if box0[2] is None else box0[2]
            self.data.qpos["box1_x"] = xyz1[0] if box1[0] is None else box1[0]
            self.data.qpos["box1_y"] = 0.001 if box1[1] is None else box1[1]
            self.data.qpos["box1_z"] = xyz1[2] if box1[2] is None else box1[2]

    def render(self, **render_kwargs):
        kwargs = render_kwargs if render_kwargs else self.render_kwargs
        return self.dm_env.physics.render(**kwargs)

    def reset(self):
        ts =  self.dm_env.reset()
        change_object_color(self,"self",self.red)
        change_object_color(self,"target",self.green)



    def step(self, a):
        return self.dm_env.step(a)

def dict_to_flat_arr(d):
    flat_arrs = np.concatenate([np.array(v).reshape(-1)
                                for v in list(d.values())]).reshape(-1)
    return np.array([v for v in flat_arrs])


def normalize_image(img):
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


def numpy_to_tensor_img(img):
    """converts numpy image to tensor

    Arguments:
        img {np.array} -- numpy array with shape (w,h,c)

    Returns:
        torch tensor -- tensor with shape (1,c,w,h)
    """
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float()
    return img.permute(0, 3, 1, 2)

def change_object_color(env,obj, color):
    _MATERIALS = [obj]

    env.model.mat_rgba[_MATERIALS] = list(
        color)+[1.0]

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
