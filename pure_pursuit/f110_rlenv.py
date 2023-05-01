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
NUM_LIDAR_SCANS = 1080

class F110Env_Continuous_Planner(gym.Env):
    def __init__(self, T=1, **kargs):
        self.T = T
        self.obs_shape = (3 + NUM_LIDAR_SCANS + self.T * 2, 1)
        
        map_name = 'levine'  # Spielberg, example, MoscowRaceway, Catalunya -- need further tuning
        try:
            map_path = os.path.abspath(os.path.join('..', 'maps', map_name))
            assert os.path.exists(map_path)
        except:
            map_path = os.path.abspath(os.path.join('maps', map_name))
        self.yaml_config = yaml.load(open(map_path + '/' + map_name + '_map.yaml'), Loader=yaml.FullLoader)

        # load waypoints
        # csv_data = np.loadtxt(map_path + '/' + map_name + '_raceline.csv', delimiter=';', skiprows=0)
        #csv_data = np.loadtxt(map_path + '/' + map_name + '_centerline.csv', delimiter=',', skiprows=0)
        csv_data = np.loadtxt(map_path + '/' + map_name + '_centerline.csv', delimiter=';', skiprows=0)

        self.main_waypoints = Waypoint(csv_data)   # process these with RL
        self.opponent_waypoints = Waypoint(csv_data)
        self.rotated_offset = [0.0, 0.0]

        # load controller
        self.main_controller = PurePursuit(self.main_waypoints)
        self.opponent_controller = PurePursuit(self.opponent_waypoints)
        self.main_renderer = Renderer(self.main_waypoints)
        self.opponent_renderer = Renderer(self.opponent_waypoints)
        self.f110 = F110Env(map=map_path + '/' + map_name + '_map', map_ext='.pgm', num_agents=2)
        # steer, speed
        
        self.action_space = spaces.Box(low=-1 * np.ones((self.T, )), high=np.ones((self.T, ))) # action ==> x-offset
        self.action_size = self.action_space.shape[0]
        # lidar_obs_shape = (NUM_LIDAR_SCANS, 1)
        low = np.concatenate((-100*np.ones((2, 1)), np.zeros((1, 1)), np.zeros((NUM_LIDAR_SCANS, 1)), np.array([-1, -1]* self.T).reshape(-1, 1))) # (lidar_scans, pointX, pointY,)
        high = np.concatenate((100*np.ones((2, 1)), 2*np.pi * np.ones((1, 1)),  1000 * np.ones((NUM_LIDAR_SCANS, 1)), np.array([1, 1]* self.T).reshape(-1, 1))) # (lidar_scans, pointX, pointY,)
        
        self.observation_space = spaces.Box(low, high, shape=self.obs_shape)
        self.reward_range = (-10, 10)
        self.metadata = {}
        self.lap_time = 0

    def reset(self, **kwargs):
        if "seed" in kwargs:
            self.seed(kwargs["seed"])
        main_agent_init_pos = np.array([self.yaml_config['init_pos']])
        opponent_init_pos = main_agent_init_pos + np.array([0, 1, 0]) # np.array([-2.4921703, -5.3199103, 4.1368272]) # TODO generate random starting point
        init_pos = np.vstack((main_agent_init_pos, opponent_init_pos))
        self.lap_time = 0
        raw_obs, _, done, _ = self.f110.reset(init_pos)
        self.prev_raw_obs = raw_obs
        obs = self._get_obs(raw_obs)
        self.prev_obs = obs
        return obs
    
    def process_action(self, action):
        pass

    def step(self, action):
        """
        action: nd.array => x-offset + original traj of length (self.T, 1)
        """
        # TODO: spline points of horizon T
        # print("action: ", action)
        R = lambda theta: np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        axis = np.array([0, 1]).reshape(-1, 1)
        # translation from
        rotated_offset = R(self.prev_raw_obs['poses_theta'][0]) @ axis * action
        # print(f"action: {action}, rotated_offset: {rotated_offset}")

        main_speed, main_steering = self.main_controller.control(obs=self.prev_raw_obs, agent=1, offset=np.array([0.2, 0.2])) #rotated_offset[:, 0])
        opponent_speed, opponent_steering = self.opponent_controller.control(obs=self.prev_raw_obs, agent=2)
        main_agent_steer_speed = np.array([[main_steering, main_speed]])
        opponent_steer_speed = np.array([[opponent_steering, opponent_speed]])

        steer_speed = np.vstack((main_agent_steer_speed, opponent_steer_speed))
        # print(steer_speed)

        '''
        at time t
        '''
        self.main_renderer.load_target_point(self.currPos, self.prev_obs, rotated_offset)
        self.f110.add_render_callback(self.main_renderer.render_point)
        raw_obs, time, done, info = self.f110.step(steer_speed)
        ''' 
        at time t+1
        '''
        self.lap_time += time
        self.prev_raw_obs = raw_obs
        obs = self._get_obs(raw_obs)
        self.prev_obs = obs
        # print(reward, info, self.f110.collisions)
        reward = -1 # control cost
        if self.f110.collisions[0] == 1:
            # print("collided: ", done, info)
            reward -= 1
        else:
            reward += 1
        # reward += self.lap_time
        
        # TODO
        # reward = self.get_reward()

        # TODO: is there anything that prevents the output of the model from given an x-offset in line? and in an approriate length?

        return obs, reward, done, info

    def _get_obs(self, raw_obs):
        obs = np.zeros(self.obs_shape)
        if isinstance(raw_obs, tuple):
            raw_obs = raw_obs[0]
        # print('agent 1: ', raw_obs['poses_x'][0], raw_obs['poses_y'][0], np.rad2deg(raw_obs['poses_theta'][0]))
        # print('agent 2: ', raw_obs['poses_x'][1], raw_obs['poses_y'][1])
        scans = raw_obs['scans'][0].reshape(-1, 1)
        # negative_distance, positive_distance = 0, 0
        # angle_increment = 360/NUM_LIDAR_SCANS
        # angles = np.deg2rad(np.arange(0, 360, angle_increment))
        # sin_angles, cos_angles = np.sin(angles), np.cos(angles)
        # negative_distance = scans[540]
        # positive_distance = scans[0]
        # print("distance: ", negative_distance, positive_distance)
        # print("sum: ", negative_distance + positive_distance)
        self.currPos = np.array([raw_obs['poses_x'][0], raw_obs['poses_y'][0], raw_obs['poses_theta'][0]])
        # xmax, xmin = self.main_waypoints.max(axis=0), self.main_waypoints.min(axis=0)
        # ymax, ymin = self.main_waypoints.max(axis=1), self.main_waypoints.min(axis=1)
        # thetamax, thetamin = 2*np.pi, 0

        obs[:3] = self.currPos.reshape(-1, 1)
        obs[3:NUM_LIDAR_SCANS+3, :] = scans
        targetPoint, idx  = self.main_controller.get_target_waypoint(self.prev_raw_obs, agent=1)
        # print("target point:", targetPoint)
        obs[-2:, :] = targetPoint.reshape(-1, 1)
        return obs
    
    def render(self, mode, **kwargs):
        self.f110.render(mode)
