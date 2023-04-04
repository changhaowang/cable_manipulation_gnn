import numpy as np

def estimate_Jacobian(robot_end_effector_pos_list, rope_history_pos_list, time_window):
    '''
    Estimate residual model Jacobian for online learning
    Args:
        1. robot_end_effector_pos_list: robot end effector position
        2. rope_history_pos_list: a list of history rope node positions
        3. time_window
    '''
    history_size = len(robot_end_effector_pos_list)
    num_nodes = len(rope_history_pos_list[0])
    robot_dim = len(robot_end_effector_pos_list[0])
    if time_window > history_size:
        J = np.zeros((robot_dim, num_nodes*2))
    else:
        A, B = prepare_matrix(robot_end_effector_pos_list, rope_history_pos_list, time_window)
        J = least_square(A, B) 
    return J

def prepare_matrix(robot_end_effector_pos_list, rope_history_pos_list, time_window):
    '''
    Helper function to prepare the least square matrices
    '''
    num_nodes = len(rope_history_pos_list[0])
    robot_dim = len(robot_end_effector_pos_list[0])
    A = np.zeros((time_window, robot_dim))
    B = np.zeros((time_window, num_nodes*2))
    for i in range(time_window):
        A[i,:] = robot_end_effector_pos_list[-i]
        B[i,:] = rope_history_pos_list[-i].flatten()
        
    return A, B

def least_square(A, B):
    return np.linalg.pinv(A.T @ A) @ A.T @ B