import numpy as np
from scipy.spatial import distance
import os


class Waypoint:
    def __init__(self, csv_data=None, centerline=False):
        if centerline:
            self.x = csv_data[:, 0]
            self.y = csv_data[:, 0]
            self.v = np.ones_like(self.y)
        else:
            self.x = csv_data[:, 1]
            self.y = csv_data[:, 2]
            self.v = csv_data[:, 5]
            self.θ = csv_data[:, 3]  # coordinate matters!f -- but pp doesn't use θ
            self.γ = csv_data[:, 4]


class PurePursuit:
    """ 
    Implement Pure Pursuit on the car
    """

    def __init__(self, waypoints):
        self.is_clockwise = False

        self._waypoints = np.array([waypoints.x, waypoints.y]).T
        self.numWaypoints = self.waypoints.shape[0]
        self.ref_speed = waypoints.v

        self.L = 1.5
        self.steering_gain = 0.5

    @property
    def waypoints(self):
        return self._waypoints
    
    def get_target_waypoint(self, obs, agent):
        # Get current pose
        self.currX = obs['poses_x'][agent - 1]
        self.currY = obs['poses_y'][agent - 1]
        self.currPos = np.array([self.currX, self.currY]).reshape((1, 2))

        # Find closest waypoint to where we are
        self.distances = distance.cdist(self.currPos, self.waypoints, 'euclidean').reshape((self.numWaypoints))
        self.closest_index = np.argmin(self.distances)

        # Find target point
        targetPoint, target_point_index = self.get_closest_point_beyond_lookahead_dist(self.L)
        # print(f"agent num: {agent} at {targetPoint}")
        #self.targetPoint = targetPoint
        return targetPoint, target_point_index

    def control(self, obs, agent, offset=None):
        print("---")
        print("agent", agent)
        # Get current pose
        self.currX = obs['poses_x'][agent - 1]
        self.currY = obs['poses_y'][agent - 1]
        self.currPos = np.array([self.currX, self.currY]).reshape((1, 2))

        # Find closest waypoint to where we are
        self.distances = distance.cdist(self.currPos, self.waypoints, 'euclidean').reshape((self.numWaypoints,))
        self.closest_index = np.argmin(self.distances)
        # print("closest_index", self.closest_index)

        # print("offset", offset)
        # Find target point
        targetPoint, target_point_index = self.get_closest_point_beyond_lookahead_dist(self.L)

        print(targetPoint)
        # print(f"agent {agent} target point {targetPoint} offset {offset}")
        # assert targetPoint.tolist() in self.waypoints.tolist(), f"{targetPoint}"
        # targetPoint = targetPoint + offset if agent == 1 else targetPoint
        if isinstance(offset, np.ndarray):  # agent == 1:
            targetPoint = targetPoint + offset  # += is not overwritten by np!
        print(targetPoint)
        # calculate steering angle / curvature
        waypoint_y = np.dot(np.array([np.sin(-obs['poses_theta'][agent - 1]), np.cos(-obs['poses_theta'][agent - 1])]),
                            targetPoint - np.array([self.currX, self.currY]))
        gamma = self.steering_gain * 2.0 * waypoint_y / self.L ** 2
        steering_angle = gamma
        # radius = 1 / (2.0 * waypoint_y / self.L ** 2)
        # steering_angle = np.arctan(0.33 / radius)  # Billy's method, but it also involves tricky fixing
        steering_angle = np.clip(steering_angle, -0.35, 0.35)

        # calculate speed
        speed = self.ref_speed[target_point_index]

        return speed, steering_angle

    def get_closest_point_beyond_lookahead_dist(self, threshold):
        point_index = self.closest_index
        print("closest_index =", self.closest_index)
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

        print("point_index =", point_index)
        point = self.waypoints[point_index]
        # print("point =", point)
        # print("waypoint[44] =", self.waypoints[44])

        return point, point_index
