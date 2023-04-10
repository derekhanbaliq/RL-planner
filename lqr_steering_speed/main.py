"""
    MEAM 517 Final Project - LQR Steering Speed Control - main application
    Author: Derek Zhou & Tancy Zhao
    References: https://f1tenth-gym.readthedocs.io/en/latest/index.html
                https://github.com/f1tenth/f1tenth_gym/tree/main/examples
                https://github.com/f1tenth/f1tenth_planning/tree/main/f1tenth_planning/control/lqr
                https://github.com/AtsushiSakai/PythonRobotics/tree/master/PathTracking/lqr_steer_control
"""

import gym
import numpy as np
import yaml
import os

from lqr_steering_speed import Waypoint, LQRSteeringSpeedController
from render import Renderer


def main():
    method_name = 'lqr_steering_speed'

    # load map & yaml
    map_name = 'MoscowRaceway'  # Spielberg, example, MoscowRaceway, Catalunya
    map_path = os.path.abspath(os.path.join('..', 'maps', map_name))
    yaml_config = yaml.load(open(map_path + '/' + map_name + '_map.yaml'), Loader=yaml.FullLoader)

    # load waypoints
    csv_data = np.loadtxt(map_path + '/' + map_name + '_raceline.csv', delimiter=';', skiprows=0)
    waypoints = Waypoint(map_name, csv_data)

    # load controller
    controller = LQRSteeringSpeedController(waypoints)

    # create env & init
    env = gym.make('f110_gym:f110-v0', map=map_path + '/' + map_name + '_map', map_ext='.png', num_agents=1)
    renderer = Renderer(waypoints)
    env.add_render_callback(renderer.render_waypoints)

    # init
    init_pos = np.array([yaml_config['init_pos']])
    obs, _, done, _ = env.reset(init_pos)
    lap_time = 0.0

    while not done:
        steering, speed = controller.control(obs)  # each agent’s current observation
        print("steering = {}, speed = {}".format(round(steering, 5), speed))

        obs, time_step, done, _ = env.step(np.array([[steering, speed]]))

        lap_time += time_step
        env.render(mode='human_fast')

    print('Sim elapsed time:', lap_time)


if __name__ == '__main__':
    main()
