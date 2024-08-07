import gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from gym.core import ObsType
from package.envs.drone import MAX_ALTITUDE, MAX_DISTANCE
import package.settings as settings
from .target_detector import TargetDetector


class CvWrapper(gym.ObservationWrapper):
    """Class responsible for target detection"""

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.detector = TargetDetector()

        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),
            high=np.array([1, 1, 1, 1, MAX_DISTANCE, MAX_ALTITUDE]),
            dtype=np.float32,
        )

        self.targets = {}

        self.distance = 0

        self.altitude = 0

    def observation(self, observation) -> ObsType:
        drone_observation = (
            observation[0] if isinstance(observation, tuple) else observation
        )

        drone_img_width = drone_observation["drone_img"].shape[0] - 1

        drone_img_height = drone_observation["drone_img"].shape[1] - 1

        self.targets = [
            list(target)
            for target in self.detector.detect(drone_observation["drone_img"])
        ]

        self.distance = drone_observation["distance"]

        self.altitude = drone_observation["altitude"]

        for target in self.targets:
            target[1] = drone_observation["drone_img"].shape[1] - target[1]
            target[3] = drone_observation["drone_img"].shape[1] - target[3]

        observations = np.array(
            [
                [target[0], target[1], target[2], target[3], target[4]]
                for target in self.targets
            ]
        )

        if len(observations):
            max_prob_target_index = np.argmax(observations[:, 4])
            max_prob_target = observations[max_prob_target_index]

            return np.array(
                [
                    max_prob_target[0] / drone_img_width,
                    max_prob_target[1] / drone_img_height,
                    max_prob_target[2] / drone_img_width,
                    max_prob_target[3] / drone_img_height,
                    drone_observation["distance"],
                    drone_observation["altitude"],
                ]
            )

        empty_observation = np.zeros(self.observation_space.shape)
        empty_observation[-2] = drone_observation["distance"]
        empty_observation[-1] = drone_observation["altitude"]

        return empty_observation

    def render(self, mode="human"):
        drone_img = super().render(mode)

        return self._add_detection_frame(drone_img)

    def _add_detection_frame(self, drone_img):
        drone_img_pil = Image.fromarray(drone_img.astype(np.uint8))
        draw = ImageDraw.Draw(drone_img_pil)

        for target in self.targets:
            xmin = target[0] if target[0] < target[2] else target[2]
            ymin = target[1] if target[1] < target[3] else target[3]
            xmax = target[0] if target[0] > target[2] else target[2]
            ymax = target[1] if target[1] > target[3] else target[3]

            top_left = (xmin, ymin)
            bottom_right = (xmax, ymax)
            rectangle_color = (255, 0, 0)  # Red color
            draw.rectangle([top_left, bottom_right], outline=rectangle_color)

        draw.text(
            (10, 10),
            f"Distance {self.distance * MAX_DISTANCE:.{3}f}",
            font=ImageFont.truetype(f"{settings.WORKING_DIRECTORY}/arial.ttf", 15),
            fill=(255, 0, 0),
        )

        draw.text(
            (10, 25),
            f"Alt {self.altitude * MAX_ALTITUDE:.{3}f}",
            font=ImageFont.truetype(f"{settings.WORKING_DIRECTORY}/arial.ttf", 15),
            fill=(255, 0, 0),
        )

        return np.array(drone_img_pil)
