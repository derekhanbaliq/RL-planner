import gym
import numpy as np
import yaml
import os
from argparse import Namespace

from pure_pursuit import PurePursuit, Waypoint
from render import Renderer
import utils


def main():
    method_name = 'pure_pursuit'

    # load map & yaml
    map_name = 'Catalunya'  # Spielberg, example, MoscowRaceway, Catalunya -- need further tuning
    map_path = os.path.abspath(os.path.join('..', 'maps', map_name))
    yaml_config = yaml.load(open(map_path + '/' + map_name + '_map.yaml'), Loader=yaml.FullLoader)

    # load waypoints
    csv_data = np.loadtxt(map_path + '/' + map_name + '_raceline.csv', delimiter=';', skiprows=0)
    main_waypoints = Waypoint(csv_data - 0.2)   # process these with RL
    opponent_waypoints = Waypoint(csv_data)

    # load controller
    main_controller = PurePursuit(main_waypoints)
    opponent_controller = PurePursuit(opponent_waypoints)

    # create env & init
    env = gym.make('f110_gym:f110-v0', map=map_path + '/' + map_name + '_map', map_ext='.png', num_agents=2)
    main_renderer = Renderer(main_waypoints)
    opponent_renderer = Renderer(opponent_waypoints)
    env.add_render_callback(main_renderer.render_waypoints)
    env.add_render_callback(opponent_renderer.render_waypoints)

    # init
    main_agent_init_pos = np.array([yaml_config['init_pos']])
    opponent_init_pos = np.array([-2.4921703, -5.3199103, 4.1368272])   # random starting point?

    init_pos = np.vstack((main_agent_init_pos, opponent_init_pos))
    obs, _, done, _ = env.reset(init_pos)
    lap_time = 0.0

    while not done:

        # RL input data for agent 1 (main car)
        scans = utils.downsample(obs['scans'], 10, 'simple')
        currX = obs['poses_x'][0]
        currY = obs['poses_y'][0]
        currTH = obs['poses_theta'][0]
        currV = obs['linear_vels_x'][0]

        # TODO: get the current waypoint -> RL planner -> new waypoints
        # wp = controller._get_current_waypoint(controller.waypoints, 5, position, pose_theta)
        
        #speed, steering = controller.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'], work['vgain'])  # planner calculate the speed & steering angle

        # plan() <-- lidar, waypoint [T],`

        # planner calculate the speed & steering angle
        main_speed, main_steering = main_controller.control(obs=obs, agent=1)
        opponent_speed, opponent_steering = opponent_controller.control(obs=obs, agent=2)

        # print("main steering = {}, main speed = {}".format(round(main_steering, 5), main_speed))

        main_agent_steer_speed = np.array([[main_steering, main_speed]])
        zero_steer_speed = np.array([[0.0, 0.0]])       # for testing purposes
        opponent_steer_speed = np.array([[opponent_steering, opponent_speed]])

        steer_speed = np.vstack((main_agent_steer_speed, opponent_steer_speed))
        obs, time_step, done, _ = env.step(steer_speed)

        lap_time += time_step
        env.render(mode='human_fast')

    print('Sim elapsed time:', lap_time)


if __name__ == '__main__':
    main()
