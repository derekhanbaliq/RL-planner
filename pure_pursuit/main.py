import gym
import numpy as np
import yaml
import os
from argparse import Namespace

from pure_pursuit import PurePursuit, Waypoint
from render import Renderer


def main():
    method_name = 'pure_pursuit'

    # load map & yaml
    map_name = 'Catalunya'  # Spielberg, example, MoscowRaceway, Catalunya
    map_path = os.path.abspath(os.path.join('..', 'maps', map_name))
    yaml_config = yaml.load(open(map_path + '/' + map_name + '_map.yaml'), Loader=yaml.FullLoader)

    # load waypoints
    csv_data = np.loadtxt(map_path + '/' + map_name + '_raceline.csv', delimiter=';', skiprows=0)
    waypoints = Waypoint(map_name, csv_data)

    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 1.0}
    # 1.375 / 0.90338203837889}, which is 8m/s for real F1TENTH car

    with open('config_map.yaml') as file:  # in current path
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)  # all parameters in yaml file

    # load controller
    # controller = PurePursuitController(conf, 0.33)
    controller = PurePursuit(waypoints)

    # create env & init
    env = gym.make('f110_gym:f110-v0', map=map_path + '/' + map_name + '_map', map_ext='.png', num_agents=1)
    renderer = Renderer(waypoints)
    env.add_render_callback(renderer.render_waypoints)

    # init
    init_pos = np.array([yaml_config['init_pos']])
    obs, _, done, _ = env.reset(init_pos)
    lap_time = 0.0

    while not done:
        # TODO: get the current waypoint -> RL planner -> new waypoints
        # wp = controller._get_current_waypoint(controller.waypoints, 5, position, pose_theta)
        
        #speed, steering = controller.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'], work['vgain'])  # planner calculate the speed & steering angle

        # plan() <-- lidar, waypoint [T],

        speed, steering = controller.control(obs)  # planner calculate the speed & steering angle
        print("steering = {}, speed = {}".format(round(steering, 5), speed))

        obs, time_step, done, _ = env.step(np.array([[steering, speed]]))

        lap_time += time_step
        env.render(mode='human_fast')

    print('Sim elapsed time:', lap_time)


if __name__ == '__main__':
    main()
