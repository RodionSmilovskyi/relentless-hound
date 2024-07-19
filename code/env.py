import numpy as np
import gymnasium as gym
import math
import pybullet as p
import pybullet_data
from PIL import Image
from gymnasium.core import ObsType, ActType
import os 


MAX_DISTANCE = 9999
FRAME_NUMBER = 100
THRUST = (0, 1)
YAW = (-1, 1)
PITCH = (-45, 45)
ROLL = (-45, 45)
DRONE_IMG_WIDTH = 256
DRONE_IMG_HEIGHT = 256
NUMBER_OF_CHANNELS = 3
THRUST_TO_WEIGHT_RATIO = 4
WORKING_DIRECTORY = os.path.dirname(os.path.abspath(__file__))


class Env(gym.Env):
    def __init__(self, useGUI = True):
        self._agent_location = np.array([0, 0, 0], dtype=np.int32)
        self.metadata = {"render_fps": 30, "render_modes": ["human", "rgb_array"]}
        self.world_space = gym.spaces.Box(
            low=np.array([-20, -20, 0]), high=np.array([20, 20, 10]), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(
            {
                "drone_img": gym.spaces.Box(
                    low=0,
                    high=DRONE_IMG_WIDTH - 1,
                    shape=(DRONE_IMG_WIDTH, DRONE_IMG_HEIGHT, NUMBER_OF_CHANNELS),
                    dtype=np.uint8,
                ),
                "distance": gym.spaces.Box(0, MAX_DISTANCE, shape=(1,)),
            }
        )
        self.action_space: ActType = gym.spaces.Box(
            low=np.array([-1, -1, -1, -1]),
            high=np.array([1, 1, 1, 1]),
            dtype=np.float32,
        )
        self.render_mode = "rgb_array"
        self._client = p.connect(p.GUI if useGUI == True else p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8, physicsClientId=self._client)

        # for i in [
        #     p.COV_ENABLE_RGB_BUFFER_PREVIEW,
        #     p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
        #     p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
        # ]:
        #     p.configureDebugVisualizer(
        #         p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1, physicsClientId=self._client
        #     )

    def step(self, action: ActType):
        def convert_range(x, x_min, x_max, y_min, y_max):
            return ((x - x_min) / (x_max - x_min)) * (y_max - y_min) + y_min
        
        MAX_MOTOR_THRUST = 2 * (THRUST_TO_WEIGHT_RATIO * 9.81) / 4
        
        thrust = convert_range(action[0], -1, 1, 0, 1)  #-1.. 1 -> 
        pitch = action[1]
        roll = action[2]
        yaw = action[3]
        
        forces = [thrust for i in range(4)]
        # forces[0] = forces[0] + pitch + roll #front l
        # forces[2] = forces[2] + pitch - roll #fron r
        # forces[1] = forces[1] - pitch  + roll #back l
        # forces[3] = forces[3] - pitch - roll # back r
        
        forces[0] = forces[0] + pitch + roll
        forces[1] = forces[1] + roll
        forces[2] = forces[2] + pitch
        
        
        for i in range(4):
            p.applyExternalForce(
                self.drone_id,
                i,
                forceObj=[0, 0, forces[i] * MAX_MOTOR_THRUST],
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
            )
        
        p.applyExternalTorque(
            self.drone_id,
            -1,  # Apply to the base
            torqueObj=[0, 0, 4 * yaw *  MAX_MOTOR_THRUST],  # Y-axis torque
            flags=p.LINK_FRAME,
        )    

        p.stepSimulation()

        self._drone_img = self._get_drone_view()
        self._dist = self._get_distance_from_sensor()

        obs: ObsType = {"drone_img": self._drone_img, "distance": self._dist}

        self._step_number = self._step_number + 1
        return obs, -1, self._step_number == FRAME_NUMBER, False, {}

    def render(self, mode="human"):
        return self._drone_img

    def close(self):
        p.disconnect()

    def reset(self, seed: int = None, options: dict = None) -> ObsType:
        p.resetSimulation()

        self._step_number = 0
        self._drone_img = np.zeros(self.observation_space["drone_img"].shape)
        self._dist = MAX_DISTANCE

        self._scene_setup()

        empty_obs: ObsType = {"drone_img": self._drone_img, "distance": self._dist}

        return empty_obs, None

    def _scene_setup(self):
        self.plane_id = p.loadURDF("plane.urdf")
        self.drone_id = p.loadURDF(f"{WORKING_DIRECTORY}/drone.urdf", basePosition=[0, 0, 1])

        self._place_target()
        self._look_at(self.drone_id, self.target_id)

    def _place_target(self):
        random_position = self.world_space.sample()
        
        self.initial_dist = np.linalg.norm(random_position - np.array([0,0,0]))
        
        while self.initial_dist < 5:
            random_position = self.world_space.sample()
            self.initial_dist = np.linalg.norm(random_position - np.array([0,0,0]))
        

        collisionShapeId = p.createCollisionShape(
            shapeType=p.GEOM_MESH, fileName=f"{WORKING_DIRECTORY}/a_cube.obj"
        )
        visualShapeId = p.createVisualShape(
            shapeType=p.GEOM_MESH, fileName=f"{WORKING_DIRECTORY}/a_cube.obj"
        )
        self.target_id = p.createMultiBody(
            baseCollisionShapeIndex=collisionShapeId,
            baseVisualShapeIndex=visualShapeId,
            basePosition=random_position.tolist(),
        )

    def _look_at(self, source_id, target_id):
        source_pos, _ = p.getBasePositionAndOrientation(source_id)
        target_pos, _ = p.getBasePositionAndOrientation(target_id)
        direction = np.array(target_pos) - np.array(source_pos)
        direction /= np.linalg.norm(direction)

        yaw = np.arctan2(direction[1], direction[0])
        pitch = np.arctan2(
            -direction[2], np.sqrt(direction[0] ** 2 + direction[1] ** 2)
        )
        roll = np.arctan2(direction[1], direction[0])
        quat = p.getQuaternionFromEuler([0, pitch, yaw])
        p.resetBasePositionAndOrientation(source_id, source_pos, quat)

    def _get_drone_view(self) -> np.array:
        pos, orn = p.getBasePositionAndOrientation(self.drone_id)
        rot_mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        target = np.dot(rot_mat, np.array([self.world_space.high[0], 0, 0])) + np.array(
            pos
        )

        drone_cam_view = p.computeViewMatrix(
            cameraEyePosition=pos, cameraTargetPosition=target, cameraUpVector=[0, 0, 1]
        )
        drone_cam_pro = p.computeProjectionMatrixFOV(
            fov=60.0, aspect=1.0, nearVal=0, farVal=np.max(self.world_space.high)
        )
        [width, height, rgbImg, dep, seg] = p.getCameraImage(
            width=256,
            height=256,
            shadow=1,
            viewMatrix=drone_cam_view,
            projectionMatrix=drone_cam_pro,
        )

        # Convert the image data to a numpy array
        image = np.array(rgbImg, dtype=np.uint8).reshape((height, width, 4))

        return image[:, :, :3]

    def _get_distance_from_sensor(self) -> np.float32:
        pos, orn = p.getBasePositionAndOrientation(self.drone_id)
        rot_mat = p.getMatrixFromQuaternion(orn)
        drone_direction = np.array([rot_mat[0], rot_mat[3], rot_mat[6]]) * MAX_DISTANCE
        ray_result = p.rayTest(pos, pos + drone_direction)
        results = [hit[2] * MAX_DISTANCE for hit in ray_result if hit[0] == self.target_id]
        
        if len(results):
            return min(results)       
        
        return MAX_DISTANCE

