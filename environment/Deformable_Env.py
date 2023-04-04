import pybullet as p
import math
import pybullet_data
import numpy as np
import cv2 as cv
from PIL import Image

class Deformable_Env(object):
    '''
    Deformable Manipulation Environment by Bullet Physics
    '''
    def __init__(self, num_robot=2, robot_init_pos_list=[[0.900000, 0.400000, 1.400000], [-0.900000, 0.400000, 1.400000]], gripper_init_pose_list=[[0.923103, -0.500000, 2.40036],[-0.923103, -0.500000, 2.40036]], vis=True, save_video = False, real_time_sim=False, fem=True, sparseSdfVoxelSize=0.25):
        '''
        Args:
            num_robot: int. Number of robots in the environment
            robot_init_pos_list: list. A list of robot initial base position in the environment
            gripper_init_pose_list: list. A list of robot initial gripper position
            vis: bool. Whether to visualize the environment
            save_video: bool. Whether to save video
            real_time_sim: bool. Whether to use real time (interactive) simulation
            fem: bool. Whether to use fem to simulate the environment
            sparseSdfVoxelSize: double. Voxel size for simulation
        '''
        if vis:
            self.physicsClient = p.connect(p.GUI) #or p.DIRECT for non-graphical version
        else:
            self.physicsClient = p.connect(p.DIRECT)
        
        if fem:
            p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # self.logId = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "visualShapeBench.json")
        
        p.setPhysicsEngineParameter(sparseSdfVoxelSize=sparseSdfVoxelSize)
        p.setRealTimeSimulation(real_time_sim)
        p.setGravity(0, 0, -9.8)

        self.num_robot = num_robot
        self.robot_init_pos_list = robot_init_pos_list
        self.gripper_init_pose_list = gripper_init_pose_list
        self.save_video = save_video
        # camera parameters
        self.pixelWidth = 320
        self.pixelHeight = 220
        self.camTargetPos = [0, -0.5, 2]
        self.camDistance = 2
        self.pitch = -40.0
        self.roll = 0
        self.yaw = 0
        self.upAxisIndex = 2
        self.viewMatrix = p.computeViewMatrixFromYawPitchRoll(self.camTargetPos, self.camDistance, self.yaw, self.pitch, self.roll,self.upAxisIndex)
        
        self.leftCoordinate = -0.6
        self.rightCoordinate = 0.6
        self.bottomCoordinate = -0.5
        self.topCoordinate = 0.5
        self.nearplaneDistance = 0.9
        self.farplaneDistance = 20
        self.projectionMatrix = p.computeProjectionMatrix(self.leftCoordinate, self.rightCoordinate, self.bottomCoordinate, self.topCoordinate, self.nearplaneDistance, self.farplaneDistance)

        self.init_env()

    def init_env(self):
        '''
        Initiaize the environment
        '''
        self.planeID = p.loadURDF("plane.urdf", [0, 0, -0.3],globalScaling=2)
        self.tableID = [p.loadURDF("table/table.urdf", [.000000, -0.00000, 0.000000], [0.000000, 0.000000,0,1],globalScaling=2)]

        # Load robot
        self.robotID_list = []
        self.gripperID_list = []
        self.robot_constraintID_list = []
        for i in range(self.num_robot):
            robotID = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], globalScaling=2)
            p.resetBasePositionAndOrientation(robotID, self.robot_init_pos_list[i], p.getQuaternionFromEuler([0,0,1.570796326794896619]))
            self.robotID_list.append(robotID)
            # Set grippers
            gripperID = p.loadSDF("gripper/wsg50_one_motor_gripper_new_free_base.sdf")
            self.gripperID_list.append(gripperID[0])
            p.resetBasePositionAndOrientation(self.gripperID_list[i], self.gripper_init_pose_list[i], [-0.000000, 0.964531, -0.000002, -0.263970])
            p.resetBasePositionAndOrientation(self.gripperID_list[i], self.gripper_init_pose_list[i],
                                        [-0.000000, 0.964531, -0.000002, -0.263970])
            gripper_jointPositions = [0.000000, -0.011130, -0.206421, 0.205143, 0.0, 0.000000, 0.0, 0.000000]
            # Set gripper pose
            for jointIndex in range(p.getNumJoints(self.gripperID_list[i])):
                p.resetJointState(self.gripperID_list[i], jointIndex, gripper_jointPositions[jointIndex])
                p.setJointMotorControl2(self.gripperID_list[i], jointIndex, p.POSITION_CONTROL, gripper_jointPositions[jointIndex], 0)
            # Add constraint
            self.robot_constraintID_list.append(p.createConstraint(self.robotID_list[i], 6, self.gripperID_list[i], 0, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0.01], [0, 0, 0]))
        # Initialize robot pose
        self.set_robot_info()

        # Load rope
        self.ropeID = p.loadSoftBody("environment/myrope.vtk", [-1,-0.5,1.300] ,p.getQuaternionFromEuler([0,1.570796326794896619,0]), mass = 8,scale = 2, \
            useMassSpring=-1, useBendingSprings=1, useNeoHookean = -1,springElasticStiffness=2000*2, springDampingStiffness=1000*2, \
            springBendingStiffness=1500*2, collisionMargin = 0.002, useSelfCollision = 1, frictionCoeff = 0.99, repulsionStiffness =1600)
        p.changeVisualShape(self.ropeID, -1, rgbaColor=[1,1,1,1], textureUniqueId=-1, flags=0)

        # step simulation a little bit 
        for i in range(100):
            self.step_sim()

        # save video
        if self.save_video:
            # Define the codec and create VideoWriter object
            fourcc = cv.VideoWriter_fourcc(*'XVID')
            self.out = cv.VideoWriter('control.avi',fourcc, 20.0, (self.pixelWidth, self.pixelHeight))

    def set_robot_info(self):
        self.num_robot_joints = p.getNumJoints(self.robotID_list[0])
        #lower limits for null space
        self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        #upper limits for null space
        self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        #joint ranges for null space
        self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        #restposes for null space
        self.rp = [0, 0, 0, 0.6 * math.pi, 0, -math.pi * 0.5 * 0.66, math.pi * 0.5]
        #joint damping coefficents
        self.jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        # Init robot pose
        for i in range(self.num_robot_joints):
            for j in range(self.num_robot):
                p.resetJointState(self.robotID_list[j], i, self.rp[i])
                p.setJointMotorControl2(self.robotID_list[j], i, p.POSITION_CONTROL, self.rp[i], 0)
            
    
    def step_sim(self):
        '''
        Step simulation and save the robot and rope info.
        '''
        p.stepSimulation()
        self.robot_info = [p.getJointStates(self.robotID_list[i], jointIndices=list(range(self.num_robot_joints))) for i in range(self.num_robot)]
        self.rope_info = p.getMeshData(self.ropeID, -1, flags=p.MESH_DATA_SIMULATION_MESH)


    def move_robot(self, desired_pose_list, desired_euler_angle_list, sim_step=50):
        ''' 
        desired end-effector (Cartesian position + euler angle) list.
        Args:
            1. desired_pose_list: list. A list of desired robot position
            2. desired_euler_angle_list: list. A list of desired euler angle
            3. sim_step: double. Number of steps to move the robot in simulation
        '''
        joint_pose_list =[]        
        for i in range(self.num_robot):
            orn = p.getQuaternionFromEuler(desired_euler_angle_list[i])
            joint_pose = p.calculateInverseKinematics(self.robotID_list[i], self.num_robot_joints-1, desired_pose_list[i], orn, self.ll, self.ul, self.jr, self.rp)
            joint_pose_list.append(joint_pose)
            for j in range(self.num_robot_joints):
                p.setJointMotorControl2(bodyIndex=self.robotID_list[i],
                                jointIndex=j,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=joint_pose[j],
                                targetVelocity=0,
                                force=500,
                                positionGain=0.05,
                                velocityGain=1)
        for i in range(sim_step):
            self.step_sim()

    def grasp_soft_object(self, gripper_id_list, object_id_list, vertices_id_list, relative_pose_list):
        '''
        Grasp soft object by creating soft constraints.
        Args:
            1. gripper_id_list: list. Gripper Ids in simulation
            2. object_id_list: list. Grasped object Id in simulation
            3. vertices_id_list: list. Index of the grasped vertices.
            4. relative_pose_list: this field is not clear from the pybullet quickguide. Just randomly set.
        '''
        soft_constraint_id_list = []
        for i in range(self.num_robot):
            for j in range(len(vertices_id_list[i])):
                soft_constraint_id = p.createSoftBodyAnchor(object_id_list[i], vertices_id_list[i][j], gripper_id_list[i], -1, relative_pose_list[i][j])

    def restart_env(self, real_time_sim=False, fem=True, sparseSdfVoxelSize=0.25):
        '''
        Reset the environment.
        Args:
            1. real_time_sim: bool. Whether to use real-time (interactice) simulation mode
            2. fem: bool. Whether to use fem
            3. sparseSdfVoxelSize: double. voxel size for fem simulation.
        '''
        # p.stopStateLogging(self.logId)
        p.resetSimulation()
        if fem:
            p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # self.logId = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "visualShapeBench.json")
        
        p.setPhysicsEngineParameter(sparseSdfVoxelSize=sparseSdfVoxelSize)
        p.setRealTimeSimulation(real_time_sim)
        p.setGravity(0, 0, -9.8)
        self.init_env()

    def calculate_rope_feature(self, rope_pose_3d):
        '''
        Obtain rope pixel position from 3d pose for baseline visual servo.
        Since visual servo utilize the pixel instead of 3d position. We transfrom the 3d position to pixel information.
        Args:
            1. rope_pose_3d: n*3 ndarray. Rope position of each node in 3D
        Returns:
            1. rope_feature_image: n*2 ndarray. Rope pixel position of each node
        '''
        rope_pose_4d = np.transpose(np.hstack((rope_pose_3d, np.ones((len(rope_pose_3d), 1))))) # x,y,z,1
        
        viewMatrix = np.transpose(np.array(self.viewMatrix).reshape([4,4]))
        projectionMatrix = np.transpose(np.array(self.projectionMatrix).reshape([4,4]))
        
        temp_rope_feature = np.dot(np.dot(projectionMatrix, viewMatrix), rope_pose_4d)
        rope_feature_image = np.array([np.array((int((temp_rope_feature[0][i]/temp_rope_feature[3][i])*int(self.pixelWidth/2))+int(self.pixelWidth/2),int((-temp_rope_feature[1][i]/temp_rope_feature[3][i])*int(self.pixelHeight/2))+int(self.pixelHeight/2))) \
                                                        for i in range(len(rope_pose_3d))]).flatten().reshape(-1,1)
        return rope_feature_image

    def camera_render(self, curr_rope_pos_3d, desired_rope_pos_3d):
        '''
        Camera render for video saving.
        Args:
            1. curr_rope_pos_3d: n*3 ndarray. Current rope position in 3D
            2. desired_rope_pose_3d: n*3 ndarray. Desired rope position in 3D
        '''
        self.img_arr = p.getCameraImage(self.pixelWidth,
                               self.pixelHeight,
                               viewMatrix=self.viewMatrix,
                               projectionMatrix=self.projectionMatrix,
                               shadow=1,
                               lightDirection=[1, 1, 1])

        self.img_arr = self.img_arr[2][:, :, [0,1,2,3]]
        image = Image.fromarray(np.array(self.img_arr))
        self.RGB_image = cv.cvtColor(np.asarray(image),cv.COLOR_RGB2BGR)
        
        curr_rope_feature_image = self.calculate_rope_feature(curr_rope_pos_3d)
        desired_rope_feature_image = self.calculate_rope_feature(desired_rope_pos_3d)
        
        # label current rope pose
        for feature_coordinate in curr_rope_feature_image.reshape(-1,2):# actual position
            cv.circle(img=self.RGB_image, center=tuple(feature_coordinate), radius=1, color=(0, 0, 255), thickness=-1) # red
        
        # label desired rope pose
        for desired_feature_coordinate in desired_rope_feature_image.reshape(-1,2):#desired position
            cv.circle(img=self.RGB_image, center=tuple(desired_feature_coordinate), radius=2, color=(255, 0, 0), thickness=-1)  # blue
        
        cv.imshow('RGB_image',self.RGB_image)
        cv.waitKey(1)
        # save video
        if self.save_video:
            self.out.write(self.RGB_image)


