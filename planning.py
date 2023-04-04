import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.append(BASE_DIR)

import pybullet as p
import time
import numpy as np
import pathlib
import copy
import ipopt
import math
import tensorflow.compat.v1 as tf
from tensorflow.contrib import predictor

from planning_control.optimizer_ipopt import MPC_Solver
from environment.Deformable_Env import Deformable_Env

from learning_to_simulate import one_step_predict_fast
from planning_control.online_learning import estimate_Jacobian

INPUT_SEQUENCE_LENGTH = 5 
#vertexID order
vertexorder=[0, 1, 2, 3, 8, 9, 10, 11, 92, 93, 94, 95, 110, 16, 35, 54, 73, 108, 17, 36, 55, 74, 18, 37, 56, 75, 19, 38, 57, 76, 20, 39, 58, 77, 21, 40, 59, 78, 22, 41, 60, 79, 23, 42, 61, 80, 24, 43, 62, 81, 44, 25, 82, 63, 26, 45, 64, 83, 102, 27, 46, 65, 84, 107, 28, 47, 66, 85, 101, 29, 48, 67, 86, 103, 30, 49, 68, 87, 104, 31, 50, 69, 88, 105, 32, 51, 70, 89, 100, 33, 52, 71, 90, 106, 34, 53, 72, 91, 109, 4, 5, 6, 7, 12, 13, 14, 15, 96, 97, 98, 99]
vertexorder_grasped_portion = vertexorder[4:-8]
vertex_downsample = vertexorder_grasped_portion[::8]
num_rope_node = len(vertex_downsample)

# Grasped point
rope_grasp_vertices_id_list = [[4, 5, 6, 7], [8, 9, 10, 11]] 
relative_pose_list = [[[0.0, 0.0, 0.65], [0.0, 0.02, 0.63], [0.0, 0.0, 0.61], [0.0, -0.02, 0.63]],[[0,0.0, 1.65], [0,0.0, 1.61], [0,0.02, 1.63], [0,-0.02, 1.63]]]
grasp_index_list = [12, 0]

def reload_latest_model():
    '''
    Load latest GNN model
    '''
    export_dir = 'saved_model'
    subdirs = [x for x in pathlib.Path(export_dir).iterdir()
           if x.is_dir() and 't            # lb[1] = 0.05' not in str(x)]
    latest = str(sorted(subdirs)[-1])
    return latest

def get_rope_velocity(grasp_vertices_list, rope_vertex_order, rope_pose, rope_pose_new):
    '''Get a list of rope relative movement for each node on the rope'''
    num_vertex = len(rope_vertex_order)
    num_robot = len(grasp_vertices_list)
    rope_velocity = np.zeros((num_vertex, 3))
    
    for i in range(num_vertex):
        vertex = rope_vertex_order[i]
        for j in range(num_robot):
            if vertex in grasp_vertices_list[j]:
                rope_velocity[i, :] = rope_pose_new[i, :] - rope_pose[i, :]
    return rope_velocity


if __name__ == "__main__":
    # Load GNN Model
    tf.disable_v2_behavior()

    data_path = 'learning_to_simulate/datasets/Rope'
    model_path = 'learning_to_simulate/models/Rope'
    
    rollout_estimator = tf.estimator.Estimator(one_step_predict_fast.get_one_step_estimator_fn(data_path, 0), model_dir=model_path)
    rollout_estimator.export_saved_model('saved_model', one_step_predict_fast.serving_input_receiver_fn)

    predict_fn = predictor.from_saved_model(reload_latest_model())

    # Open Pybullet Simulation Environment
    env = Deformable_Env(vis=True)
    max_outer_run_step = 1000

    max_inner_run_step = 1000
    run_step = 0

    grasped = 0
    soft_constraint_created = 0
    robot_grasp_pose_list = np.array([[0.98000, -0.5, 1.56], [-0.98000, -0.5, 1.56]])
    robot_grasp_euler_angle_list = np.array([[math.pi, 0, 0.5*math.pi], [math.pi, 0, 0.5*math.pi]])

    # Set a pre-defined or random desired rope pose
    rope_desired_pose = np.array([[-0.2392314 , -0.11612185,  1.39444251],
       [-0.19644994, -0.09794318,  1.36952182],
       [-0.07999273, -0.00364237,  1.32945861],
       [ 0.10893297,  0.06129222,  1.2844609 ],
       [ 0.26571585, -0.07170155,  1.2733658 ],
       [ 0.23632585, -0.27779912,  1.26270759],
       [ 0.13652451, -0.4567417 ,  1.252     ],
       [ 0.09613635, -0.65275508,  1.28321589],
       [ 0.0815559 , -0.80206257,  1.26602395],
       [ 0.15285637, -0.93858611,  1.26466536],
       [ 0.31999653, -1.00663014,  1.35539347],
       [ 0.42045375, -1.03320327,  1.36712   ],
       [ 0.61406568, -0.99035276,  1.43340171]])
    rope_desired_pose = rope_desired_pose[:,0:2]

    robot_curr_pose_list = copy.deepcopy(robot_grasp_pose_list)
    combined_robot_end_effector_pos_list = []
    rope_pos_error_list = [np.zeros((len(vertex_downsample), 2))]

    weight_GNN = 1
    error = 100 # initialize error
    mse_error = error**2
    mse_threshold = 0.05

    while run_step < max_inner_run_step and mse_error >= mse_threshold:
        robot_end_effector_pose_list = [p.getLinkState(env.robotID_list[i], env.num_robot_joints-1)[0] for i in range(env.num_robot)]
        robot_end_effector_euler_angle_list = [p.getEulerFromQuaternion(p.getLinkState(env.robotID_list[i], env.num_robot_joints-1)[1]) for i in range(env.num_robot)]
        dist = sum([np.linalg.norm(np.array(robot_end_effector_pose_list[i]) - np.array(robot_grasp_pose_list[i])) for i in range(env.num_robot)])/2
        if dist >= 0.05 and not soft_constraint_created:
            env.move_robot(robot_grasp_pose_list, robot_grasp_euler_angle_list)
        else:
            grasped += 1
        if grasped == 1:
            env.grasp_soft_object(env.gripperID_list, [env.ropeID, env.ropeID], rope_grasp_vertices_id_list, relative_pose_list)
            soft_constraint_created = 1
        
        if soft_constraint_created:
            mesh=p.getMeshData(env.ropeID, -1, flags=p.MESH_DATA_SIMULATION_MESH)
            rope_curr_pose = np.array([mesh[1][i] for i in vertex_downsample])
            if run_step == 0:
                # Initialize history data list
                rope_history_pose_list = [rope_curr_pose[:,0:2]] * (INPUT_SEQUENCE_LENGTH - 1)
                rope_history_control_list = []
            
            rope_history_pose_list.append(rope_curr_pose[:,0:2])
            combined_robot_1_end_effector_pos = np.hstack((robot_end_effector_pose_list[0][0:2], robot_end_effector_euler_angle_list[0][-1]))
            combined_robot_2_end_effector_pos = np.hstack((robot_end_effector_pose_list[1][0:2], robot_end_effector_euler_angle_list[1][-1]))
            combined_robot_end_effector_pos = np.hstack((combined_robot_1_end_effector_pos, combined_robot_2_end_effector_pos))
            combined_robot_end_effector_pos_list.append(combined_robot_end_effector_pos)

            # Estimate Jacobian
            time_window = 10
            J = estimate_Jacobian(combined_robot_end_effector_pos_list, rope_pos_error_list, time_window)
            # Optimization
            rope_pose_dim = 2 # x. y
            robot_control_dim = 3 # x, y, w
            horizon = 5
            # Initial condition
            lb = [-0.1] * robot_control_dim * 2 * horizon
            ub = [0.1] * robot_control_dim * 2 * horizon
            x0 = np.zeros((robot_control_dim*horizon*2, ))
            # Solve NLP
            nlp = ipopt.problem(
                n=len(x0),
                m=0,
                problem_obj=MPC_Solver(rope_desired_pose, rope_history_pose_list[-5:], predict_fn, grasp_index_list, rope_pose_dim, robot_control_dim, horizon, weight_GNN, J),
                lb=lb,
                ub=ub,
                )
            nlp.addOption('mu_strategy', 'adaptive')
            nlp.addOption('tol', 1e-3)
            nlp.addOption('max_iter', 10)
            nlp.addOption('print_level', 0)
            start = time.time()
            x, info = nlp.solve(x0)
            end = time.time()
            # print('Solve time', end - start)
            # Move the robot
            robot_curr_pose_list = [p.getLinkState(env.robotID_list[i], env.num_robot_joints-1)[0] for i in range(env.num_robot)]
            robot_curr_euler_angle_list = [p.getEulerFromQuaternion(p.getLinkState(env.robotID_list[i], env.num_robot_joints-1)[1]) for i in range(env.num_robot)]
            
            robot_next_pose_list = [robot_curr_pose_list[i] + np.hstack((x[i*robot_control_dim:i*robot_control_dim+2],0)) for i in range(2)]
            for j in range(2):
              robot_next_pose_list[j][2] = 1.71 # set z to be a constant

            robot_next_euler_angle_list = [robot_curr_euler_angle_list[i] + np.hstack((np.zeros((2,)), x[i*robot_control_dim     + 2])) for i in range(2)]

            # input('Press Any Key to move the robot!!!!')
            env.move_robot(robot_next_pose_list, robot_next_euler_angle_list)
            robot_curr_pose_list = robot_next_pose_list
            run_step += 1
            mesh=p.getMeshData(env.ropeID, -1, flags=p.MESH_DATA_SIMULATION_MESH)
            rope_new_pose = np.array([mesh[1][i] for i in vertex_downsample])
            rope_pos_error_list.append(rope_new_pose[:,0:2] - rope_curr_pose[:,0:2])
            
            if error <= np.linalg.norm(rope_desired_pose - rope_new_pose[:,0:2]):
                weight_GNN *= 0.95
            error =np.linalg.norm(rope_desired_pose - rope_new_pose[:,0:2])
            mse_error = error ** 2
            print('mean squared error', mse_error)
