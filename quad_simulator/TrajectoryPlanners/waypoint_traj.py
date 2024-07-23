import numpy as np
from numpy import linalg


class WaypointTraj(object):
    def __init__(self, points, desired_speed):
        self.points         = points
        self.desired_spd    = desired_speed

        num_pts     = self.points.shape[0]
        dist        = linalg.norm(np.diff(self.points, axis=0), axis=1)
        self.time   = np.zeros((num_pts,))
        for i, d in enumerate(dist):
            # v = 0.05 * np.log(d) + 0.25
            # if v < 0.01:
            #     v = 0.0
            v = 1.5
            self.time[i + 1] = self.time[i] + d/v

    def update(self, t):
        x = np.zeros((3,))
        x_dot = np.zeros((3,))

        x_ddot = np.zeros((3,))
        x_dddot = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        num_segments = len(self.points) - 1
        if num_segments > 0:
            segment_dists = self.points[1:(num_segments + 1), :] - self.points[0:num_segments, :]
            norm_dists = np.linalg.norm(segment_dists, axis=1)
            unit_vec = segment_dists / norm_dists[:, None]
            segment_times = norm_dists / self.desired_spd
            start_times = np.cumsum(segment_times)

            if t < start_times[len(start_times) - 1]:
                idx = np.where(t <= start_times)[0]
                segment_num = idx[0]

                diff_time = t - start_times[segment_num]
                x_dot = self.desired_spd * unit_vec[segment_num, :]
                if x.any() <= self.points[segment_num + 1, :].any():
                    x = self.points[segment_num + 1, :] + x_dot * (diff_time/1)
                else:
                    x = self.points[segment_num + 1, :]
            else:  # time exceeds expected time at last waypoint
                segment_num = num_segments - 1
                x_dot = np.zeros((3,))
                x = self.points[segment_num + 1, :]
        else:
            x_dot = np.zeros((3,))
            x = self.points

        output = np.hstack((x, x_dot, x_ddot, x_dddot, x_ddddot, yaw, yaw_dot))
        return output