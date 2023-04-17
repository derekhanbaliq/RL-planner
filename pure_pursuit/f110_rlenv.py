# -- coding: utf-8 --
import gym
from gym import spaces

import os
import yaml
import numpy as np
from scipy.spatial import distance

from f110_gym.envs.f110_env import F110Env
from pure_pursuit import PurePursuit, Waypoint
from render import Renderer


class F110Env_Continuous_Planner(gym.Env):
    def __init__(self, T=1, **kargs):
        self.T = T
        self.obs_shape = (1080 + self.T * 2, 1)
        
        map_name = 'Catalunya'  # Spielberg, example, MoscowRaceway, Catalunya -- need further tuning
        try:
            map_path = os.path.abspath(os.path.join('..', 'maps', map_name))
            assert os.path.exists(map_path)
        except:
            map_path = os.path.abspath(os.path.join('maps', map_name))
        self.yaml_config = yaml.load(open(map_path + '/' + map_name + '_map.yaml'), Loader=yaml.FullLoader)

        # load waypoints
        csv_data = np.loadtxt(map_path + '/' + map_name + '_raceline.csv', delimiter=';', skiprows=0)
        main_waypoints = Waypoint(csv_data)   # process these with RL
        opponent_waypoints = Waypoint(csv_data)

        # load controller
        self.main_controller = PurePursuit(main_waypoints)
        self.opponent_controller = PurePursuit(opponent_waypoints)
        self.main_renderer = Renderer(main_waypoints)
        self.opponent_renderer = Renderer(opponent_waypoints)
        self.f110 = F110Env(map=map_path + '/' + map_name + '_map', map_ext='.png', num_agents=2)
        # steer, speed
        
        self.action_space = spaces.Box(low=-1 * np.ones((self.T, )), high=np.ones((self.T, ))) # , shape=(1,))
        self.action_size = self.action_space.shape[0]
        lidar_obs_shape = self.obs_shape[0] - self.T * 2
        low = np.concatenate((np.zeros((lidar_obs_shape, 1)), np.array([-1, -1]* self.T).reshape(-1, 1))) # (lidar_scans, pointX, pointY,)
        high = np.concatenate((1000 * np.ones((lidar_obs_shape, 1)), np.array([1, 1]* self.T).reshape(-1, 1))) # (lidar_scans, pointX, pointY,)
        
        self.observation_space = spaces.Box(low, high, shape=self.obs_shape)
        self.reward_range = (-1, 1)
        self.metadata = {}

    def reset(self, **kwargs):
        if "seed" in kwargs:
            self.seed(kwargs["seed"])
        main_agent_init_pos = np.array([self.yaml_config['init_pos']])
        opponent_init_pos = np.array([-2.4921703, -5.3199103, 4.1368272]) # random starting point?
        init_pos = np.vstack((main_agent_init_pos, opponent_init_pos))
        raw_obs, _, done, _ = self.f110.reset(init_pos)
        obs = self._get_obs(raw_obs)
        self.prev_raw_obs = raw_obs
        return obs

    def step(self, action):
        """
        action: nd.array => original traj of llength (self.T, 1)
        """

        main_speed, main_steering = self.main_controller.control(obs=self.prev_raw_obs, agent=1)
        opponent_speed, opponent_steering = self.opponent_controller.control(obs=self.prev_raw_obs, agent=2)
        main_agent_steer_speed = np.array([[main_steering, main_speed]])
        opponent_steer_speed = np.array([[opponent_steering, opponent_speed]])

        steer_speed = np.vstack((main_agent_steer_speed, opponent_steer_speed))

        raw_obs, reward, done, info = self.f110.step(steer_speed)
        obs = self._get_obs(raw_obs)
        self.prev_obs = obs
        self.prev_raw_obs = raw_obs

        # TODO
        # reward = self.get_reward() 

        return obs, reward, done, info

    def _get_obs(self, raw_obs):
        obs = np.zeros(self.obs_shape)
        if isinstance(raw_obs, tuple):
            raw_obs = raw_obs[0]
        obs[:1080, :] = raw_obs['scans'][0].reshape(-1, 1)

        # self.main_controller
        self.main_controller.currX = raw_obs['poses_x'][0]
        self.main_controller.currY = raw_obs['poses_y'][0]
        self.main_controller.currPos = np.array([self.main_controller.currX, self.main_controller.currY]).reshape((1, 2))
        self.main_controller.distances = distance.cdist(self.main_controller.currPos, self.main_controller.waypoints, 'euclidean').reshape((self.main_controller.numWaypoints))
        self.main_controller.closest_index = np.argmin(self.main_controller.distances)

        # Find target point
        targetPoint, _ = self.main_controller.get_closest_point_beyond_lookahead_dist(self.main_controller.L)

        obs[-2:, :] = targetPoint.reshape(-1, 1)
        return obs
    
    def render(self, mode, **kwargs):
        self.f110.render(mode)
