from dm_control import suite
import numpy as np
from gym.spaces import Box, Dict
import os
import torch
from vqvae.models.vqvae import VQVAE
from .point_mass import DMImageGoalPointMassEnv, DMPointMassEnv
from vqvae.planner import RepresentationGraph, return_shortest_path
import numpy as np
from vqvae import utils


class DMImageGoalPointMassEnvWithVQVAE(DMImageGoalPointMassEnv):

    def __init__(self,
                 graph_file=None,
                 model_filename=None,
                 model_dir=None,
                 min_rep_count=100,
                 experiment='her',
                 gpu_id=0,
                 latent_dim=64,
                 goal_dim=64,
                 build_graph=False,
                 ** kwargs):

        DMImageGoalPointMassEnv.__init__(self, **kwargs)
        self.gpu_id = gpu_id
        self.device = torch.device(
            "cuda:" + str(self.gpu_id) if torch.cuda.is_available() else "cpu")

        if model_filename is None:
            raise ValueError(
                'Must supply a valid path to VQVAE model .pth file')
        # the experiment name
        self.experiment = experiment
        # loads vq vae model
        self.model, self.model_params = self._load_model(
            model_filename, model_dir)

        if build_graph:
            self._build_graph(graph_file)

        self._set_obseravtion_space(latent_dim, goal_dim)

    def _set_obseravtion_space(self, latent_dim, goal_dim):
        """
        defines observation as space dict object
        """
        # obs and goal spaces are Boxes, each has the size of the latent state vector
        self.reps_state_space = Box(
            low=-float('inf'), high=float('inf'), shape=(latent_dim,), dtype=np.float32)
        self.goal_space = Box(
            low=-float('inf'), high=float('inf'), shape=(goal_dim,), dtype=np.float32)
        self.image_space = Box(low=0, high=255, shape=(
            self.img_dim, self.img_dim, self.num_channels))

        self.observation_space = Dict([
            ('observation', self.reps_state_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.goal_space),
            ('state_observation', self.reps_state_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.goal_space),
            ('image_desired_goal', self.image_space),
            ('image_achieved_goal', self.image_space),
        ])

    def _build_graph(self, graph_file):
        """
        loads representation graph or creates one of it  doesn'y exist
        """
        if graph_file is None:
            rep_graph = RepresentationGraph(model_filename=model_filename,
                                            model_dir=model_dir,
                                            min_rep_count=100)
            self.graph = rep_graph.graph
            self.rep_dict = rep_graph.rep_dict
            self.rep_to_state = rep_graph.rep_dict.rep_to_state
            self.hash_to_rep = rep_graph.rep_dict.hash_to_rep
        else:
            rep_graph = np.load(graph_file, allow_pickle=True)
            self.graph = rep_graph.item().get('graph')
            self.rep_dict = rep_graph.item().get('rep_dict')
            self.rep_to_state = rep_graph.item().get('rep_to_state')
            self.hash_to_rep = rep_graph.item().get('hash_to_rep')

    def _load_model(self, model_filename, model_dir=None):
        """
        Loads the VQ VAE model
        if running from the VQ VAE folder, you can set model_dir = None
        """

        path = os.getcwd() + '/results/' if model_dir is None else model_dir

        if torch.cuda.is_available():
            data = torch.load(path + model_filename)
        else:
            data = torch.load(path+model_filename,
                              map_location=lambda storage, loc: storage)

        params = data["hyperparameters"]

        model = VQVAE(params['n_hiddens'], params['n_residual_hiddens'],
                      params['n_residual_layers'], params['n_embeddings'],
                      params['embedding_dim'], params['beta'], gpu_id=self.gpu_id).to(self.device)

        model.load_state_dict(data['model'])

        return model, params

    def reset(self):
        """
        resets the environment, allows HER and PLAN type experiments
        """
        if 'her' in self.experiment:
            # get original goal obs from env
            goal_obs = DMImageGoalPointMassEnv.reset(self)
            # encode images to get representations
            achieved_rep = self._get_normalized_encoding(
                "image_achieved_goal", goal_obs, 8.0)
            self.desired_rep = self._get_normalized_encoding(
                "image_desired_goal", goal_obs, 8.0)
            # set representations as achieved / desired goals & state
            goal_obs = self._set_goal_obs(
                goal_obs, achieved_rep, self.desired_rep)
            # current rep is whatever we just achieved
            self.current_rep = achieved_rep
        else:
            raise NotImplementedError(
                'Only HER is supported, but if you want to use planner then comment this error.')
            goal_obs = self._reset_to_state_from_graph()

        return goal_obs

    def _get_normalized_encoding(self, image_key, goal_obs, dim):
        rep = self._encode_image(goal_obs[image_key])
        return (rep - dim)/dim

    def _set_goal_obs(self, goal_obs, achieved_rep, desired_rep):
        goal_obs['state_desired_goal'] = desired_rep
        goal_obs['state_achieved_goal'] = achieved_rep
        goal_obs['state_observation'] = achieved_rep
        goal_obs['desired_goal'] = desired_rep
        goal_obs['achieved_goal'] = achieved_rep
        goal_obs['observation'] = achieved_rep
        return goal_obs

    def compute_reward(self, action, obs, *args, **kwargs):
        """
        her - 0 if representation is exactly equal to desired one
        plan - 0 if representation is exactly equal to next node
        fuzzy_her - (todo) 0 if representation is approximately equaly to desired one
        """
        if 'her' in self.experiment:
            if np.linalg.norm(self.current_rep - self.desired_rep, 2) == 0:
                return 0.
            else:
                return - 1.
        elif 'fuzzy_her':
            raise NotImplementedError('fuzzy_her has not been implemented')
        elif 'plan' in self.experiment:
            if np.linalg.norm(self.current_rep - self.next_node_rep, 2) == 0:
                return 0.
            else:
                return -1.

    def step(self, action, debug=False):
        if debug:
            start = time.time()

        obs_next, reward, done, info = DMPointMassEnv.step(self, action)
        reward = self.compute_reward(None, None)
        # done should be determined by representation, not threshold
        if 'her_naive' not in self.experiment:

            if np.linalg.norm(self.current_rep - self.desired_rep, 2) == 0:
                done = True
                info['is_success'] = self.env_step
            else:
                done = False
                info['is_success'] = 0
            if self.env_step >= self.max_steps:
                done = True

        # encode image to get rep and normalize
        goal_obs_next = DMImageGoalPointMassEnv._obs_to_goal_obs(
            self, obs_next)
        achieved_rep = self._get_normalized_encoding(
            "image_achieved_goal", goal_obs_next, 8.0)
        # set current rep (remember it's normalized)
        self.current_rep = achieved_rep
        # also re-compute reward
        reward = self.compute_reward(None, None)

        # set normalized achieved rep here
        goal_obs_next = self._set_goal_obs(
            goal_obs_next, achieved_rep, self.desired_rep)

        # if plan, check reward to transition to next node in graph

        if 'plan' in self.experiment:
            if reward == 0.0 and len(self.path) > 0:
                next_node_hash = self.pop(0)
                self.next_node_rep = self.hash_to_rep[next_node_hash]
                self.next_node_rep = (self.next_node_rep - 8.0)/8.0
            goal_obs_next['state_desired_goal'] = self.next_node_rep
            goal_obs_next['desired_goal'] = self.next_node_rep

        if debug:
            print('Time per step', time.time()-start)
            print('Size of obs obj', getsizeof(goal_obs_next))

        return goal_obs_next, reward, done, info

    def _reset_her_experiment(self):
        goal_obs = DMImageGoalPointMassEnv.reset(self)

    def _reset_to_state_from_graph(self):
        obs = DMPointMassEnv.reset(self)

        def get_rand(d):
            return np.random.choice(list(d.keys()))

        path = False

        while not path:
            start_rep_hash, end_rep_hash = get_rand(
                self.rep_dict), get_rand(self.rep_dict)
            path = return_shortest_path(
                self.graph, start_rep_hash, end_rep_hash)
            # if path:
            #    if len(path) < 3:
            #        path = False

        # the starting state should not be in the path
        # since it's already achieved
        self.path = path[1:]
        # set next state
        next_node_hash = self.path.pop(0)
        self.next_node_rep = self.hash_to_rep[next_node_hash]
        start_rep = self.hash_to_rep[start_rep_hash]
        self.desired_rep = self.hash_to_rep[end_rep_hash]

        # 1. set aux rep as path[1]
        # 2. give 0 for reaching any rep in graph, -1 otherwise
        # (3.) if rep is in forward path of graph, skip to it (optional)
        # 4. each time node is reached move on to next node
        # 5. once no more nodes episode ends

        goal_state = self.rep_to_state[end_rep_hash]
        node_state = self.rep_to_state[next_node_hash]
        start_state = self.rep_to_state[start_rep_hash]

        # set simulator to goal state
        # get image obs

        self.dm_env.physics.named.data.geom_xpos['pointmass'] = goal_state
        goal_img = self.render(self.img_dim, self.img_dim)

        # set simulator to node state
        self.dm_env.physics.named.data.geom_xpos['pointmass'] = node_state
        node_img = self.render(self.img_dim, self.img_dim)

        # set simulator to start state
        # get image obs

        self.dm_env.physics.named.data.geom_xpos['pointmass'] = start_state
        start_img = self.render(self.img_dim, self.img_dim)

        # set the goal obs dict

        goal_obs = {}
        self.desired_rep = (self.desired_rep - 8.0) / 8.0
        self.next_node_rep = (self.next_node_rep - 8.0)/8.0
        start_rep = (start_rep-8.0)/8.0
        goal_obs['state_desired_goal'] = self.next_node_rep
        goal_obs['state_achieved_goal'] = start_rep
        goal_obs['state_observation'] = start_rep
        goal_obs['desired_goal'] = self.next_node_rep
        goal_obs['achieved_goal'] = start_rep
        goal_obs['observation'] = start_rep
        goal_obs['image_achieved_goal'] = start_rep
        goal_obs['image_desired_goal'] = goal_img
        # define this later
        # goal_obs['image_node_goal'] = node_img
        self.desired_goal_image = goal_img

        self.current_rep = start_rep

        return goal_obs

    def _encode_image(self, img):
        img = np.array([img]).transpose(0, 3, 1, 2)
        img = torch.tensor(img).float()
        img = img.to(self.device)
        vq_encoder_output = self.model.pre_quantization_conv(
            self.model.encoder(img))

        encoder_mean = -2.855
        encoder_std = 8.83
        _, _, _, _, e_indices = self.model.vector_quantization(
            vq_encoder_output)
        return e_indices.cpu().detach().numpy().squeeze()


if __name__ == "__main__":
    # model_dir = '/home/misha/downloads/vqvae/results/'
    # model_filename = 'vqvae_data_point_mass_v2ne16nd16.pth'
    # graph_file = model_dir + 'vqvae_graph_point_mass_v2ne16nd16.npy'
    model_dir = '/home/misha/downloads/vqvae/results/'
    model_filename = 'vqvae_data_point_mass_jul3_ne16nd16.pth'
    graph_file = '/home/misha/downloads/vqvae/results/jul8_graph.npy'

    print('loading graph and model')
    env = DMImageGoalPointMassEnvWithVQVAE(graph_file=graph_file,
                                           model_dir=model_dir,
                                           model_filename=model_filename,
                                           experiment='her_naive'
                                           )
    obs = env.reset()
    # print('reset obs', obs.keys())
    for _ in range(505):
        obs_, r, d, info = env.step(env.action_space.sample())
    for k, v in obs_.items():
        print(k, ':', v)
    print('reward', r)
    print('done', d)
    # print('next obs', obs_.keys())
    # print('reward', r)
    # print('distance to target', env._distance_to_target())
    # print('img shape', env.render(32, 32).shape)

    # print('achieved', (obs_['state_achieved_goal']-8.)/16.)

    # print('desired', obs_['state_desired_goal'])
