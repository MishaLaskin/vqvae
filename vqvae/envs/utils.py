from dm_control import suite
from gym.spaces import Box, Dict
import numpy as np
from functools import reduce
from vqvae.models.vqvae import VQVAE, TemporalVQVAE
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
