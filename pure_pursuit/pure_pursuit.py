import numpy as np
from scipy.spatial import distance
import os

class Waypoint:
    def __init__(self, map_name, csv_data=None):
        if map_name == 'Spielberg' or map_name == 'MoscowRaceway' or map_name == 'Catalunya':
            self.x = csv_data[:, 1]
            self.y = csv_data[:, 2]
            self.v = csv_data[:, 5]
            self.θ = csv_data[:, 3]  # coordinate matters!
            self.γ = csv_data[:, 4]
        elif map_name == 'example' or map_name == 'icra':
            self.x = csv_data[:, 1]
            self.y = csv_data[:, 2]
            self.v = csv_data[:, 5]
            self.θ = csv_data[:, 3] + np.pi / 2  # coordinate matters!
            self.γ = csv_data[:, 4]


class PurePursuit:
    """ 
    Implement Pure Pursuit on the car
    """
    def __init__(self, waypoints):
        self.is_clockwise = True
        self.waypoints = np.array([waypoints.x, waypoints.y]).T

        self.numWaypoints = self.waypoints.shape[0]
        self.ref_speed = waypoints.v * 0.06 #0.6

        self.L = 2.2
        self.steering_gain = 0.1 #0.45

    def control(self, obs):

        # ['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'], work['vgain']
        
        # Get current pose
        self.currX = obs['poses_x'][0]
        self.currY = obs['poses_y'][0]
        self.currPos = np.array([self.currX, self.currY]).reshape((1, 2))

        print(self.currPos.shape)
        print(self.waypoints.shape)
        # Find closest waypoint to where we are
        self.distances = distance.cdist(self.currPos, self.waypoints, 'euclidean').reshape((self.numWaypoints))
        self.closest_index = np.argmin(self.distances)
        self.closestPoint = self.waypoints[self.closest_index]

        # Find target point
        targetPoint = self.get_closest_point_beyond_lookahead_dist(self.L)
        
        # calculate curvature/steering angle
        y = targetPoint[1]
        gamma = self.steering_gain * (2 * y / self.L**2)
        steering_angle = np.clip(gamma, -0.35, 0.35)

        # calculate speed
        speed = self.ref_speed[self.closest_index]

        print("steering = {}, speed = {}".format(round(steering_angle, 2), round(speed, 2)))

        return speed, steering_angle
        
    def get_closest_point_beyond_lookahead_dist(self, threshold):
        point_index = self.closest_index
        dist = self.distances[point_index]
        
        while dist < threshold:
            if self.is_clockwise:
                point_index -= 1
                if point_index < 0:
                    point_index = len(self.waypoints) - 1
                dist = self.distances[point_index]
            else:
                point_index += 1
                if point_index >= len(self.waypoints):
                    point_index = 0
                dist = self.distances[point_index]

        point = self.waypoints[point_index]

        return point
