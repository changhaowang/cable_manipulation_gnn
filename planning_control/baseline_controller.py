import numpy as np


class Baseline_Controller(object):
    '''
    Visual servoing baseline controller based on https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6631329
    Contact: yuyouz@andrew.cmu.edu for more details
    '''
    def __init__(self, alpha, k, J, dim, desired_rope_feature):
        self.jacobian = J
        self.k = k
        self.alpha = alpha
        self.dim = dim
        self.desired_rope_feature = desired_rope_feature

    def update_jacobian(self, curr_rope_feature, prev_rope_feature, curr_end_effector_pos, prev_end_effector_pos):
        self.curr_rope_feature = np.array(curr_rope_feature)
        self.curr_end_effector_pos = curr_end_effector_pos

        delta_feature = np.array(np.array(curr_rope_feature) - np.array(prev_rope_feature)).flatten().reshape(-1,1)
        delta_end_effector_pos = curr_end_effector_pos - prev_end_effector_pos
        self.jacobian = self.jacobian + np.dot(self.alpha * (delta_feature - np.dot(self.jacobian, delta_end_effector_pos))/np.dot(delta_end_effector_pos.reshape(1,-1),delta_end_effector_pos), delta_end_effector_pos.reshape(1,-1))

    def get_robot_control(self):
        num_features = len(self.desired_rope_feature)
        G = np.identity(int(num_features))
        feature_error = self.curr_rope_feature - self.desired_rope_feature
        control = -np.dot(self.k , np.dot(np.linalg.pinv(self.jacobian), np.dot(G, feature_error.reshape(-1,1))))
        return control.flatten()