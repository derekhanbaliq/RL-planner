"""
    MEAM 517 Final Project - LQR Steering Speed Control - Renderer class
    Author: Derek Zhou & Tancy Zhao
    References: https://github.com/f1tenth/f1tenth_gym/tree/main/examples
"""
import numpy as np
from pyglet.gl import GL_POINTS  # game interface


class Renderer:

    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.drawn_waypoints = []
        self.obs = None
        self.prev_points = [0.0, 0.0]

    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """
        points = np.vstack((self.waypoints.x, self.waypoints.y)).T  # N x 2

        scaled_points = 50. * points

        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [255, 255, 255]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]

    def render_point(self, e):
        """
        draw one point
        """
        tp = self.target_points * 50.0
        # erase previous target points
        e.batch.add(1, GL_POINTS, None, ('v3f/stream', [self.prev_points[0], self.prev_points[1], 0.]), ('c3B/stream', [4, 30, 91]))
        e.batch.add(1, GL_POINTS, None, ('v3f/stream', [self.prev_points[0] + 1, self.prev_points[1], 0.]), ('c3B/stream', [4, 30, 91]))
        e.batch.add(1, GL_POINTS, None, ('v3f/stream', [self.prev_points[0], self.prev_points[1] + 1, 0.]), ('c3B/stream', [4, 30, 91]))
        e.batch.add(1, GL_POINTS, None, ('v3f/stream', [self.prev_points[0] - 1, self.prev_points[1], 0.]), ('c3B/stream', [4, 30, 91]))
        e.batch.add(1, GL_POINTS, None, ('v3f/stream', [self.prev_points[0], self.prev_points[1] - 1, 0.]), ('c3B/stream', [4, 30, 91]))
        e.batch.add(1, GL_POINTS, None, ('v3f/stream', [self.prev_points[0] + 1, self.prev_points[1] + 1, 0.]), ('c3B/stream', [4, 30, 91]))
        e.batch.add(1, GL_POINTS, None, ('v3f/stream', [self.prev_points[0] - 1, self.prev_points[1] - 1, 0.]), ('c3B/stream', [4, 30, 91]))
        e.batch.add(1, GL_POINTS, None, ('v3f/stream', [self.prev_points[0] - 1, self.prev_points[1] + 1, 0.]), ('c3B/stream', [4, 30, 91]))
        e.batch.add(1, GL_POINTS, None, ('v3f/stream', [self.prev_points[0] + 1, self.prev_points[1] - 1, 0.]), ('c3B/stream', [4, 30, 91]))

        # plot target point
        e.batch.add(1, GL_POINTS, None, ('v3f/stream', [tp[0], tp[1], 0.]),('c3B/stream', [255, 0, 0]))
        e.batch.add(1, GL_POINTS, None, ('v3f/stream', [tp[0]+1, tp[1], 0.]), ('c3B/stream', [255, 0, 0]))
        e.batch.add(1, GL_POINTS, None, ('v3f/stream', [tp[0], tp[1]+1, 0.]), ('c3B/stream', [255, 0, 0]))
        e.batch.add(1, GL_POINTS, None, ('v3f/stream', [tp[0]-1, tp[1], 0.]), ('c3B/stream', [255, 0, 0]))
        e.batch.add(1, GL_POINTS, None, ('v3f/stream', [tp[0], tp[1]-1, 0.]), ('c3B/stream', [255, 0, 0]))
        e.batch.add(1, GL_POINTS, None, ('v3f/stream', [tp[0]+1, tp[1]+1, 0.]), ('c3B/stream', [255, 0, 0]))
        e.batch.add(1, GL_POINTS, None, ('v3f/stream', [tp[0]-1, tp[1]-1, 0.]), ('c3B/stream', [255, 0, 0]))
        e.batch.add(1, GL_POINTS, None, ('v3f/stream', [tp[0]-1, tp[1]+1, 0.]), ('c3B/stream', [255, 0, 0]))
        e.batch.add(1, GL_POINTS, None, ('v3f/stream', [tp[0]+1, tp[1]-1, 0.]), ('c3B/stream', [255, 0, 0]))

        # plot current pose
        #e.batch.add(1, GL_POINTS, None, ('v3f/stream', [self.current_pose[0] * 50.0, self.current_pose[1] * 50.0, 0.]),('c3B/stream', [255, 255, 255]))
        self.prev_points = tp

    def load_target_point(self, current_pose, target_points, offsets):
        self.target_points = target_points[-2:,:] + offsets
        self.current_pose = current_pose


    def load_obs(self, obs):
        self.obs = obs

    def render_path(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """
        x = self.obs['poses_x']
        y = self.obs['poses_y']

        point = np.array([x, y])

        scaled_point = 50. * point

        b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_point[0], scaled_point[1], 0.]),
                        ('c3B/stream', [255, 0, 0]))
        self.drawn_waypoints.append(b)
