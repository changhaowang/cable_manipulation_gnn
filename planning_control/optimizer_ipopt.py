import copy
import ipopt
import numpy as np
import pybullet as p
import tensorflow.compat.v1 as tf
from tensorflow.contrib import predictor

from learning_to_simulate import one_step_predict_fast


INPUT_SEQUENCE_LENGTH = 5

def recover_robot_control(num_rope_node, robot_control_dim, grasp_vertex_index_list, optimization_var):
    '''
    Recover the robot control from the input data format. (A reverse function of get_robot_control in collect_rope_data_2d.py)
    Args:
        1. num_rope_node: number of nodes on the rope
        2. robot_control_dim: robot degrees of freedom
        3. grasp_vertex_index_list: index of the grasped node on the rope.
        4. optimization_var: optimization variable used in the ipopt solver.
    '''
    robot_control = np.zeros((num_rope_node, robot_control_dim))
    for i in range(len(grasp_vertex_index_list)):
        index = grasp_vertex_index_list[i]
        robot_control[index,:] = optimization_var[i*robot_control_dim:(i+1)*robot_control_dim]
    return robot_control

def generate_GNN_input(num_rope_node, rope_history_pose_list, robot_curr_control, rope_pose_dim, robot_control_dim):
    '''
    Geneaate the desired GNN input format from the corresponding data.
    Args:
        1. num_rope_node: number of nodes on the rope
        2. rope_history_pose_list: history of rope position
        3. robot_curr_control: robot control in the desired format
        4. rope_pose_dim: dimesion of the rope pose (2d/3d)
        5. robot_control_dim: dimension of the robot control (2/3) 
    '''
    rope_pose_sequence = np.zeros((num_rope_node, INPUT_SEQUENCE_LENGTH, rope_pose_dim))
    robot_control_sequence = np.zeros((num_rope_node, INPUT_SEQUENCE_LENGTH, robot_control_dim))
    for i in range(INPUT_SEQUENCE_LENGTH):
        rope_pose_sequence[:,i,:] = rope_history_pose_list[i]
    robot_control_sequence[:,INPUT_SEQUENCE_LENGTH-1,:] = robot_curr_control # wrong, convert it to the same format as in the training data
    input_dict = {}
    input_dict['position'] = rope_pose_sequence.astype(np.float32)
    input_dict['velocity'] = robot_control_sequence.astype(np.float32)
    return input_dict

def get_final_rope_prediction(robot_control, rope_history_pose_list, rope_model, horizon, rope_pose_dim, robot_control_dim, grasp_index_list, weight_GNN, J):
    '''
    Rollout learned model by 'horizon' steps to get final rope prediction
    Args:
        1. robot_control: robot control matrix in the desired format
        2. rope_history_pose_list: rope history positions
        3. rope_model: learned GNN model
        4. horizon: number of steps of MPC
        5. rope_pose_dim: rope position dimension
        6. robot_control_dim: robot control dimension
        7. grasp_index_list: index of grasped nodes on the rope
        8. weight_GNN: weight to use the offline learned GNN
        9. J: online learned local linear residual model
    Return:
        1. rope_pose: final predicted rope position
    '''
    robot_control_matrix = np.reshape(robot_control, (horizon, robot_control_dim * 2))
    num_rope_node = len(rope_history_pose_list[0])
    rope_pose_list = copy.deepcopy(rope_history_pose_list)
    for i in range(horizon):
        robot_curr_control = robot_control_matrix[i,:]
        robot_curr_control_GNN = recover_robot_control(num_rope_node, robot_control_dim, grasp_index_list, robot_curr_control)
        input_dict = generate_GNN_input(num_rope_node, rope_pose_list[-5:], robot_curr_control_GNN, rope_pose_dim, robot_control_dim)
        online_linear_model = rope_pose_list[-1] + (np.transpose(J) @ robot_curr_control.reshape((robot_control_dim*2,-1))).reshape((-1, rope_pose_dim)) 
        prediction = rope_model(input_dict)
        rope_next_pose = weight_GNN * prediction['predicted_rollout'][0,0,:,:] + (1 - weight_GNN) * online_linear_model
        rope_pose_list.append(rope_next_pose)
    return rope_pose_list[-1]

class MPC_Solver(object):
    '''
    MPC solver for online rope control
    '''
    def __init__(self, rope_desired_pose, rope_history_pose_list, rope_model, grasp_index_list, rope_pose_dim, robot_control_dim, optimizer_horizon, weight_GNN=0.5, J=0):
        self.rope_desired_pose = rope_desired_pose
        self.num_rope_node = rope_desired_pose.shape[0]
        self.rope_history_pose_list = rope_history_pose_list
        self.rope_model = rope_model
        self.grasp_index_list = grasp_index_list
        self.rope_pose_dim = rope_pose_dim
        self.robot_control_dim = robot_control_dim
        self.horizon = optimizer_horizon
        self.J = J # online estimated jacobian
        self.weight_GNN = weight_GNN
        self.weight_online = 1 - self.weight_GNN
        self.rope_weight_matrix = np.eye(self.num_rope_node)
        # change node weight
        self.rope_weight_matrix[0,0] = 0
        self.rope_weight_matrix[-1,-1] = 0
        for i in range(3, self.num_rope_node-3):
            self.rope_weight_matrix[i,i] = 1.5

    def objective(self, x):
        '''
        Objective: matcht the final shape to the desired
        '''
        robot_control_matrix = np.reshape(x, (self.horizon, self.robot_control_dim * 2))
        rope_pose_list = copy.deepcopy(self.rope_history_pose_list)
        difference = 0
        
        for i in range(self.horizon):
            robot_curr_control = robot_control_matrix[i,:]
            robot_curr_control_GNN = recover_robot_control(self.num_rope_node, self.robot_control_dim, self.grasp_index_list, robot_curr_control)
            input_dict = generate_GNN_input(self.num_rope_node, rope_pose_list[-5:], robot_curr_control_GNN, self.rope_pose_dim, self.robot_control_dim)
            online_linear_model = rope_pose_list[-1] + (np.transpose(self.J) @ robot_curr_control.reshape((self.robot_control_dim*2,-1))).reshape((-1, self.rope_pose_dim)) 
            prediction = self.rope_model(input_dict)
            rope_next_pose = self.weight_GNN * prediction['predicted_rollout'][0,0,:,:] + self.weight_online * online_linear_model
            rope_pose_list.append(rope_next_pose)
            difference += np.linalg.norm(self.rope_weight_matrix @ (rope_next_pose - self.rope_desired_pose)) ** 2
        return difference

    def gradient(self, x):
        '''
        Utilize finite difference to approximate the gradient of the objective function
        '''
        delta = 1e-3
        gradient_list = np.zeros((self.horizon*self.robot_control_dim*2,))
        for i in range(len(x)):
            e = np.zeros((self.horizon * self.robot_control_dim * 2, ))
            e[i] = delta
            x_upper = x + e
            x_lower = x - e
            obj_upper = self.objective(x_upper)
            obj_lower = self.objective(x_lower)
            gradient = (obj_upper - obj_lower) / (2*delta)
            gradient_list[i] = gradient
        return gradient_list
