import time
from typing import Any, Tuple
import gym
from gym.core import ObsType, ActType
import numpy as np
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client as bc
from package import settings

DRONE_IMG_WIDTH = 256
DRONE_IMG_HEIGHT = 256
NUMBER_OF_CHANNELS = 3
MAX_DISTANCE = 40  # meters
MAX_ALTITUDE = 121  # meters
MIN_ALTITUDE = 1  # meters
FRAME_NUMBER = 500
THRUST_TO_WEIGHT_RATIO = 4
DRONE_WEIGHT = 1
G = 9.81


def convert_range(
    x: float, x_min: float, x_max: float, y_min: float, y_max: float
) -> float:
    """Converts value from one range system to another"""
    return ((x - x_min) / (x_max - x_min)) * (y_max - y_min) + y_min


class DroneEnv(gym.Env):
    """Class responsible for drone avionics"""

    def __init__(self, use_gui=True) -> None:
        self.plane_id = None

        self.drone_id = None

        self.target_id = None

        self.step_number = 0

        self.use_gui = use_gui

        self.metadata = {"render_fps": 30, "render_modes": ["human", "rgb_array"]}

        self._agent_location = np.array([0, 0, 0], dtype=np.int32)

        self.world_space = gym.spaces.Box(
            low=np.array([-20, -20, 0]), high=np.array([20, 20, 10]), dtype=np.float32
        )

        self.observation_space: ObsType = gym.spaces.Dict(
            {
                "drone_img": gym.spaces.Box(
                    low=0,
                    high=DRONE_IMG_WIDTH - 1,
                    shape=(DRONE_IMG_WIDTH, DRONE_IMG_HEIGHT, NUMBER_OF_CHANNELS),
                    dtype=np.uint8,
                ),
                "altitude": gym.spaces.Box(0, MAX_ALTITUDE, shape=(1,)),
                "distance": gym.spaces.Box(0, MAX_DISTANCE, shape=(1,)),
            }
        )

        self.action_space: ActType = gym.spaces.Box(
            low=np.array([-1, -1, -1, -1]),
            high=np.array([1, 1, 1, 1]),
            dtype=np.float32,
        )

        self.drone_img = np.zeros(self.observation_space["drone_img"].shape)

        self.render_mode = "rgb_array"

        # pylint: disable=c-extension-no-member
        self.client = bc.BulletClient(
            connection_mode=p.GUI if self.use_gui is True else p.DIRECT
        )

        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.client.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)

        self.client.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

        self.client.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)

    def reset(
        self,
        *,
        seed: int | None = None,
        return_info: bool = False,
        options: dict | None = None,
    ) -> ObsType:
        self.client.resetSimulation()

        self.client.setGravity(0, 0, -G)

        self.drone_img = np.zeros(self.observation_space["drone_img"].shape)

        self.step_number = 0

        self.plane_id = self.client.loadURDF("plane.urdf")

        random_position = self.world_space.sample()

        initial_dist = np.linalg.norm(random_position - np.array([0, 0, 0]))

        while initial_dist < 5:
            random_position = self.world_space.sample()
            initial_dist = np.linalg.norm(random_position - np.array([0, 0, 0]))

        collision_shape_id = self.client.createCollisionShape(
            shapeType=self.client.GEOM_MESH,
            fileName=f"{settings.WORKING_DIRECTORY}/a_cube.obj",
        )
        visual_shape_id = self.client.createVisualShape(
            shapeType=self.client.GEOM_MESH,
            fileName=f"{settings.WORKING_DIRECTORY}/a_cube.obj",
        )
        self.target_id = self.client.createMultiBody(
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=random_position.tolist(),
        )

        self.drone_id = self.client.loadURDF(
            f"{settings.WORKING_DIRECTORY}/drone.urdf",
            [0, 0, 1],
            self._look_at([0, 0, 1], random_position),
        )

        return {
            "drone_img": self.drone_img,
            "distance": 1,
            "altitude": 0,
        }, {}

    # pylint: disable=c-extension-no-member
    def step(self, action: Any) -> Tuple[ObsType, float, bool, dict]:
        self._apply_physics(action)

        altitude = self._get_altitude()

        distance = self._get_distance()

        self.drone_img = self._get_drone_view()

        self.step_number = self.step_number + 1

        low_altitude_penalty = (
            -10
            if altitude < MIN_ALTITUDE / MAX_ALTITUDE and self.step_number > 50
            else 0
        )

        reward = low_altitude_penalty

        return (
            {
                "drone_img": self.drone_img,
                "distance": distance,
                "altitude": altitude,
            },
            reward,
            self.step_number == FRAME_NUMBER or low_altitude_penalty != 0,
            {},
        )

    def render(self, mode="human"):
        return self.drone_img

    def close(self):
        self.client.disconnect()

    def _apply_physics(self, action: ActType):
        max_motor_thrust = DRONE_WEIGHT * THRUST_TO_WEIGHT_RATIO * G / 4
        thrust = convert_range(action[0], -1, 1, 0, 1)
        roll = action[1]
        pitch = action[2]
        yaw = action[3]

        forces = [max_motor_thrust * thrust for i in range(4)]

        forces[0] = forces[0] + roll * max_motor_thrust - pitch * max_motor_thrust
        forces[1] = forces[1] + roll * max_motor_thrust + pitch * max_motor_thrust
        forces[2] = forces[2] - roll * max_motor_thrust - pitch * max_motor_thrust
        forces[3] = forces[3] - roll * max_motor_thrust + pitch * max_motor_thrust

        forces = [
            forces[i] if forces[i] < max_motor_thrust else max_motor_thrust
            for i in range(4)
        ]

        for i in range(4):
            self.client.applyExternalForce(
                self.drone_id,
                i,
                forceObj=[0, 0, forces[i]],
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
            )

        self.client.applyExternalTorque(
            self.drone_id,
            -1,  # Apply to the base
            torqueObj=[0, 0, yaw * max_motor_thrust * 4],  # Y-axis torque
            flags=p.LINK_FRAME,
        )

        self.client.stepSimulation()

        if self.use_gui is True:
            time.sleep(0.01)

        motor_positions = [
            [1.2, 1.2, 0],  # Motor 1
            [-1.2, 1.2, 0],  # Motor 2
            [1.2, -1.2, 0],  # Motor 3
            [-1.2, -1.2, 0],  # Motor 4
        ]

        for ind, pos in enumerate(motor_positions):
            start_pos = pos
            end_pos = pos[0], pos[1], pos[2] - forces[ind]

            # Add the debug line
            p.addUserDebugLine(
                start_pos, end_pos, [1, 0, 0], 2, parentObjectUniqueId=self.drone_id
            )  #

    def _get_distance(self) -> float:
        pos, orn = self.client.getBasePositionAndOrientation(self.drone_id)
        rot_mat = self.client.getMatrixFromQuaternion(orn)
        drone_direction = np.array([rot_mat[0], rot_mat[3], rot_mat[6]]) * MAX_DISTANCE
        ray_result = self.client.rayTest(pos, pos + drone_direction)
        results = [hit[2] for hit in ray_result if hit[0] != -1]

        p.addUserDebugLine(
            [0, 0, 0],
            [MAX_DISTANCE, 0, 0],
            [0, 1, 0],
            2,
            0.1,
            parentObjectUniqueId=self.drone_id,
        )

        if len(results):
            return min(results)

        return 1

    def _get_altitude(self) -> float:
        pos, _ = self.client.getBasePositionAndOrientation(self.drone_id)

        return pos[2] / MAX_ALTITUDE

    def _get_drone_view(self) -> np.array:
        pos, orn = self.client.getBasePositionAndOrientation(self.drone_id)
        rot_mat = np.array(self.client.getMatrixFromQuaternion(orn)).reshape(3, 3)
        target = np.dot(rot_mat, np.array([self.world_space.high[0], 0, 0])) + np.array(
            pos
        )

        drone_cam_view = self.client.computeViewMatrix(
            cameraEyePosition=pos, cameraTargetPosition=target, cameraUpVector=[0, 0, 1]
        )
        drone_cam_pro = self.client.computeProjectionMatrixFOV(
            fov=60.0, aspect=1.0, nearVal=0, farVal=np.max(self.world_space.high)
        )
        [width, height, rgb_img, dep, seg] = self.client.getCameraImage(
            width=256,
            height=256,
            shadow=1,
            viewMatrix=drone_cam_view,
            projectionMatrix=drone_cam_pro,
        )

        # Convert the image data to a numpy array
        image = np.array(rgb_img, dtype=np.uint8).reshape((height, width, 4))

        return image[:, :, :3]

    def _look_at(self, source_pos, target_pos):
        direction = np.array(target_pos) - np.array(source_pos)
        direction /= np.linalg.norm(direction)

        yaw = np.arctan2(direction[1], direction[0])
        pitch = np.arctan2(
            -direction[2], np.sqrt(direction[0] ** 2 + direction[1] ** 2)
        )

        quat = self.client.getQuaternionFromEuler([0, pitch, yaw])
        return quat
