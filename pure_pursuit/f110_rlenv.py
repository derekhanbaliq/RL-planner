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
import utils
NUM_LIDAR_SCANS = 1080//10
SCAN_MAX = 30
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

        # load controller
        self.main_controller = PurePursuit(self.main_waypoints)
        self.opponent_controller = PurePursuit(self.opponent_waypoints)
        self.main_renderer = Renderer(self.main_waypoints)
        self.opponent_renderer = Renderer(self.opponent_waypoints)
        self.f110 = F110Env(map=map_path + '/' + map_name + '_map', map_ext='.pgm', num_agents=1)
        # steer, speed
        
        self.action_space = spaces.Box(low=-1 * np.ones((self.T, )), high=np.ones((self.T, ))) # action ==> x-offset
        # self.action_space = spaces.Discrete(9, start=-4)
        # self.action_space = spaces.MultiDiscrete(9*np.ones((self.T, )))

        low = np.concatenate((-1*np.ones((2, 1)), np.zeros((1, 1)), np.zeros((NUM_LIDAR_SCANS, 1)), np.array([-1, -1]* self.T).reshape(-1, 1))) # (lidar_scans, pointX, pointY,)
        high = np.concatenate((1*np.ones((2, 1)), 1 * np.ones((1, 1)),  1 * np.ones((NUM_LIDAR_SCANS, 1)), np.array([1, 1]* self.T).reshape(-1, 1))) # (lidar_scans, pointX, pointY,)
        
        self.observation_space = spaces.Box(low, high, shape=self.obs_shape)
        self.reward_range = (-10, 10)
        self.metadata = {}
        self.lap_time = 0
        self.prev_action = None
        self.action_diff_penalty = 1

    def reset(self, **kwargs):
        if "seed" in kwargs:
            self.seed(kwargs["seed"])
        main_agent_init_pos = np.array([self.yaml_config['init_pos']])
        # opponent_init_pos = main_agent_init_pos + np.array([0, 1, 0]) # np.array([-2.4921703, -5.3199103, 4.1368272]) # TODO generate random starting point
        # init_pos = np.vstack((main_agent_init_pos, opponent_init_pos))
        self.lap_time = 0
        self.prev_action = None
        raw_obs, _, done, _ = self.f110.reset(main_agent_init_pos)
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
        if self.T > 1:
            control_action = action[0]
        elif self.T == 1:
            control_action = action
        else:
            raise IndexError("T <= 0")
        if isinstance(self.action_space, gym.spaces.Discrete):
            action -= 4
            action /= 4
        R = lambda theta: np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        axis = np.array([0, 1]).reshape(-1, 1)
        # control_action = 0.0
        # print("action: ", action)
        rotated_offset = R(self.prev_raw_obs['poses_theta'][0]) @ axis * control_action
        # print(f"action: {action}, rotated_offset: {rotated_offset[:, 0]}")

        main_speed, main_steering = self.main_controller.control(obs=self.prev_raw_obs, agent=1, offset=rotated_offset[:, 0])
        # opponent_speed, opponent_steering = self.opponent_controller.control(obs=self.prev_raw_obs, agent=2)
        main_agent_steer_speed = np.array([[main_steering, main_speed]])
        # opponent_steer_speed = np.array([[opponent_steering, opponent_speed]])

        # steer_speed = np.vstack((main_agent_steer_speed, opponent_steer_speed))
        # print(steer_speed)

        raw_obs, time, done, info = self.f110.step(main_agent_steer_speed)
        self.lap_time += time
        self.prev_raw_obs = raw_obs
        obs = self._get_obs(raw_obs)
        self.prev_obs = obs
        # print(reward, info, self.f110.collisions)
        reward = -1 # control cost
        if self.f110.collisions[0] == 1:
            # print("collided: ", done, info)
            reward -= 100
        else:
            reward += 1
        reward += self.lap_time # np.exp(-self.lap_time)/1000
        action_diff = np.abs(action[:-1] - action[1:])
        reward -= self.action_diff_penalty * np.sum(action_diff)

        
        # TODO
        # reward = self.get_reward()

        # TODO: is there anything that prevents the output of the model from given an x-offset in line? and in an approriate length?

        return obs, reward, done, info

    def _get_obs(self, raw_obs):
        obs = np.zeros(self.obs_shape)
        if isinstance(raw_obs, tuple):
            raw_obs = raw_obs[0]
        # scans = raw_obs['scans'][0].reshape(-1, 1)
        # negative_distance, positive_distance = 0, 0
        # angle_increment = 360/NUM_LIDAR_SCANS
        # angles = np.deg2rad(np.arange(0, 360, angle_increment))
        # sin_angles, cos_angles = np.sin(angles), np.cos(angles)
        # negative_distance = scans[540]
        # positive_distance = scans[0]
        # print("distance: ", negative_distance, positive_distance)
        # print("sum: ", negative_distance + positive_distance)
        currPos = np.array([raw_obs['poses_x'][0], raw_obs['poses_y'][0], raw_obs['poses_theta'][0]])
        currPos[:1] /= 100
        currPos[-1] /= (2*np.pi)
        obs[:3] = currPos.reshape(-1, 1)
        scans = utils.downsample(raw_obs['scans'][0], NUM_LIDAR_SCANS, 'simple')
        scans = np.clip(scans, 0, SCAN_MAX)
        scans /= SCAN_MAX
        obs[3:NUM_LIDAR_SCANS+3, :] = scans.reshape(-1, 1)
        def normalize(pt, min, max):
            pt = (pt - min)/(max - min)
            pt = pt * 2 - 1
            return pt
        
        targetPoint, idx  = self.main_controller.get_target_waypoint(self.prev_raw_obs, agent=1)
        targetPoint[0] = normalize(targetPoint[0], self.main_controller.waypoints[:, 0].min(), self.main_controller.waypoints[:, 0].max())
        targetPoint[1] = normalize(targetPoint[1], self.main_controller.waypoints[:, 1].min(), self.main_controller.waypoints[:, 1].max())
        for i in range(self.T):
            start_idx = NUM_LIDAR_SCANS+3+i*2
            end_idx = start_idx + 2
            if i == 0:
                obs[start_idx:end_idx, :] = targetPoint.reshape(-1, 1)
            else:
                wp = self.main_controller.waypoints[idx + i]
                wp[0] = normalize(wp[0], self.main_controller.waypoints[:, 0].min(), self.main_controller.waypoints[:, 0].max())
                wp[1] = normalize(wp[1], self.main_controller.waypoints[:, 1].min(), self.main_controller.waypoints[:, 1].max())
                obs[start_idx:end_idx, :] = wp.reshape(-1, 1)
        return obs
    
    def render(self, mode, **kwargs):
        self.f110.render(mode)
