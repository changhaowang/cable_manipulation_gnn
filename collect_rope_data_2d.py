"""
This file is used for collecting the rope movement data by randomly move the robot end-effectors
"""

import pybullet as p
import copy

import math
import numpy as np
from  itertools import chain
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

from environment.Deformable_Env import Deformable_Env

# Global information
# vertex index of each point on the rope in sequence (from left to right)
vertexorder=[0, 1, 2, 3, 8, 9, 10, 11, 92, 93, 94, 95, 110, 16, 35, 54, 73, 108, 17, 36, 55, 74, 18, 37, 56, 75, 19, 38, 57, 76, 20, 39, 58, 77, 21, 40, 59, 78, 22, 41, 60, 79, 23, 42, 61, 80, 24, 43, 62, 81, 44, 25, 82, 63, 26, 45, 64, 83, 102, 27, 46, 65, 84, 107, 28, 47, 66, 85, 101, 29, 48, 67, 86, 103, 30, 49, 68, 87, 104, 31, 50, 69, 88, 105, 32, 51, 70, 89, 100, 33, 52, 71, 90, 106, 34, 53, 72, 91, 109, 4, 5, 6, 7, 12, 13, 14, 15, 96, 97, 98, 99]
vertexorder_grasped_portion = vertexorder[4:-8]

vertex_downsample = vertexorder_grasped_portion[::8]
print('Number of Nodes:', len(vertex_downsample))

# Grasped point by each robot. The first list shows the grasped point index of the right robot. The second shows the grasped point index of the left robot
rope_grasp_vertices_id_list = [[4, 5, 6, 7], [8, 9, 10, 11]] 
relative_pose_list = [[[0.0, 0.0, 0.65], [0.0, 0.02, 0.63], [0.0, 0.0, 0.61], [0.0, -0.02, 0.63]],[[0,0.0, 1.65], [0,0.0, 1.61], [0,0.02, 1.63], [0,-0.02, 1.63]]] # not useful

tfrecords_file = 'learning_to_simulate/datasets/Rope/Debug.tfrecord' # data folder (create test/train.tfrecord coresspondingly)
writer = tf.io.TFRecordWriter(tfrecords_file)

def get_robot_control(grasp_vertices_list, rope_vertex_order, rope_pose, rope_pose_new, euler_angle_random_disturb_list):
    '''
    Get a list of rope relative movement for each node on the rope.
    This function is purely used for save the movement of the robot into the desired data format as required for training the GNN
    Args:
        1. grasp_vertices_list: a list of grasped point index
        2. rope_vertex_order: a list of rope vertex order in sequence
        3. rope pose: n*2 ndarray. Position of each point of the rope
        4. rope_pose_new: n*2 ndarray. New position of each point on the rope
        5. euler_angle_random_distrub_list: movement of euler angle on the grapsed point
    Returns:
        1. robot_control: desried format for training the GNN
    '''
    num_vertex = len(rope_vertex_order)
    num_robot = len(grasp_vertices_list)
    robot_control = np.zeros((num_vertex, 3))
    
    for i in range(num_vertex):
        vertex = rope_vertex_order[i]
        for j in range(num_robot):
            if vertex in grasp_vertices_list[j]:
                robot_control[i, :] = rope_pose_new[i, :] - rope_pose[i, :]
                robot_control[i, -1] = euler_angle_random_disturb_list[j]
    return robot_control
            

if __name__ == "__main__":
    env = Deformable_Env(vis=True)
    max_outer_run_step = 1000 # collect 1000 trajectories
    counter = 0
    num_data_saved = 0

    while counter < max_outer_run_step:
        counter += 1
        max_inner_run_step = 50 # each trajectory contains 50 rope poses
        run_step = 0

        grasped = 0
        soft_constraint_created = 0
        desired_pose_list = np.array([[0.98000, -0.5, 1.56], [-0.98000, -0.5, 1.56]]) # here we also want to control the robot third euler angle
        desired_euler_angle_list = np.array([[math.pi, 0, 0.5*math.pi], [math.pi, 0, 0.5*math.pi]])
        rope_pose_stack = np.array([])
        robot_control_stack = np.array([])

        while run_step < max_inner_run_step:
            robot_end_effector_pose_list = [p.getLinkState(env.robotID_list[i], env.num_robot_joints-1)[0] for i in range(env.num_robot)]
            robot_end_effecotr_euler_angle_list = [p.getEulerFromQuaternion(p.getLinkState(env.robotID_list[i], env.num_robot_joints-1)[1]) for i in range(env.num_robot)]
            dist = sum([np.linalg.norm(np.array(robot_end_effector_pose_list[i]) - np.array(desired_pose_list[i])) for i in range(env.num_robot)])/2
            if dist >= 0.05 and not soft_constraint_created:
                env.move_robot(desired_pose_list, desired_euler_angle_list) # move the robot down until contact.
            else:
                grasped+= 1
            if grasped == 1:
                env.grasp_soft_object(env.gripperID_list, [env.ropeID, env.ropeID], rope_grasp_vertices_id_list, relative_pose_list) # After the robot contacts with the rope, create soft constraints
                soft_constraint_created = 1
            
            if soft_constraint_created:
                desired_new_pose_list = copy.deepcopy(desired_pose_list)
                desired_new_euler_angle_list = copy.deepcopy(desired_euler_angle_list)
                # if the rope is grasped, then randomly move the robot to collect rope pose data
                pose_random_disturb_list = []
                euler_angle_random_disturb_list = []
                for i in range(env.num_robot):
                    pose_random_disturb = np.random.uniform(-0.15, 0.15, [2,])
                    euler_angle_random_disturb = np.random.uniform(-math.pi/15, math.pi/15)
                    
                    pose_random_disturb_list.append(pose_random_disturb)
                    euler_angle_random_disturb_list.append(euler_angle_random_disturb)
                    
                    desired_new_pose_list[i][0:2] = desired_new_pose_list[i][0:2] + pose_random_disturb
                    desired_new_pose_list[i][2] = 1.71
                    desired_new_euler_angle_list[i][-1] += euler_angle_random_disturb

                dist = np.linalg.norm(desired_new_pose_list[0] - desired_new_pose_list[1])
                
                if dist <= 1*2: # sometiems the simulation may not be stable. Only save data when the simulation is stable
                    desired_pose_list = desired_new_pose_list
                    desired_euler_angle_list = desired_new_euler_angle_list

                    mesh=p.getMeshData(env.ropeID, -1, flags=p.MESH_DATA_SIMULATION_MESH)
                    rope_pose = np.array([mesh[1][i] for i in vertex_downsample])
                    rope_pose_2d = rope_pose[:,0:2]
                    # Concatenated Pose
                    rope_pose_stack = np.hstack((rope_pose_stack, rope_pose_2d.flatten()))
                    # Move the robot
                    env.move_robot(desired_pose_list, desired_euler_angle_list)
                    mesh=p.getMeshData(env.ropeID, -1, flags=p.MESH_DATA_SIMULATION_MESH)
                    rope_pose_new = np.array([mesh[1][i] for i in vertex_downsample])
                    # Calculate rope relative movement as the velocity (Only calculate the vertex that grasped by the robot)
                    if run_step != max_inner_run_step - 1:                    
                        robot_control = get_robot_control(rope_grasp_vertices_id_list, vertex_downsample, rope_pose, rope_pose_new, euler_angle_random_disturb_list)
                    else:
                        robot_control = np.zeros((len(vertex_downsample), 3))
                    
                    robot_control_stack = np.hstack((robot_control_stack, robot_control.flatten()))
                    run_step += 1

        # Save data to tfrecord
        if all(abs(rope_pose_stack) < 5):
            num_data_saved += 1
            feature_rope_pose = [tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(v).encode('utf-8') for v in rope_pose_stack]))]
            feature_robot_control = [tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(v).encode('utf-8') for v in robot_control_stack]))]
            feature_list = {'position': tf.train.FeatureList(feature=feature_rope_pose), 'velocity': tf.train.FeatureList(feature=feature_robot_control)}

            context_feature = {'key': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(num_data_saved)])), 'particle_type': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(5).encode('utf-8') for i in range(len(vertex_downsample))]))}
            example = tf.train.SequenceExample(
                context=tf.train.Features(feature=context_feature),
                feature_lists=tf.train.FeatureLists(
                feature_list=feature_list)
                )
            serialied = example.SerializeToString()
            writer.write(serialied)
            print('Scenario: '+str(counter)+', Data Saved. Number of Saved Data: '+str(num_data_saved))
        else:
            print('Scenario: '+str(counter)+', Data Not Saved!!!!')
        # Reset the simulation and restart
        env.restart_env()
            
    p.disconnect()
