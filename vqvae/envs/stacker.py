from dm_control import suite
from gym.spaces import Box, Dict
import numpy as np
from functools import reduce
from vqvae.models.vqvae import VQVAE, TemporalVQVAE
from vqvae import utils
import torch

class StackerLatentGoalEnv:

    def __init__(self, obs_dim=128, goal_dim=128,
                 env_name='stacker', task='stack_2_blocks',
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

        self.block_env = RefTwoBlocksEnv()

        self.red = np.array([246.,91.,75.])/255.
        self.green = np.array([93.,205.,189.])/255.

    def reset(self):
        """ reset and get state obs, state ag, and state dg """
        ts = self.dm_env.reset()
        #change_object_color(self,"self",self.green)
        change_object_color(self,"target",self.red)

        self.steps = 0
        hand_pos = self.data.geom_xpos["hand"].copy()
        box_pos = self.data.geom_xpos["box0"].copy()
        second_box_pos = self.data.geom_xpos["box1"].copy()
        
        self.goal = np.array([second_box_pos[0], 0.001, self.box_size*3])
        flat_obs = dict_to_flat_arr(ts.observation)
        flat_obs = np.append(flat_obs, [self.goal[0], self.goal[2]])

        """ get image and encode it to get obs """
        img = self.render(**self.render_kwargs)
        obs_z = self.img_to_latent(img, self.model1)

        """ use block env to set desired goal encoding and encode achieved goal """
        self.block_env.reset()
        # desired goal
        self.block_env.set_block_pos(box0=[self.goal[0],None,self.goal[2]], box1=second_box_pos)
        goal_img = self.block_env.render(**self.render_kwargs)
        self.goal_image = goal_img

        goal_z = self.img_to_latent(goal_img, self.model2)
        self.latent_goal = goal_z

        # achieved goal
        self.block_env.set_block_pos(box0=box_pos, box1=second_box_pos)
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
        
        #change_object_color(self,"self",self.green)
        change_object_color(self,"target",self.red)



    def step(self, a):
        return self.dm_env.step(a)


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
