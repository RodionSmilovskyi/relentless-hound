import gym
from gym.core import ActType

DISTANCE_TO_TARGET = 0.5  # meters


def is_target_at_the_center(bbox):
    """Function to check if target bb is centered"""
    x1, y1, x2, y2 = bbox
    center_x, center_y = 0.5, 0.5
    x_min = min(x1, x2)
    x_max = max(x1, x2)
    y_min = min(y1, y2)
    y_max = max(y1, y2)
    return x_min <= center_x <= x_max and y_min <= center_y <= y_max


class CenterRewardWrapper(gym.wrappers.FrameStack):
    """Class responsible for assigning rewards"""

    def __init__(self, env: gym.Env):
        super().__init__(env, 4)

        self.prev_area = 0

    def step(self, action: ActType):
        frames, reward, terminated, info = super().step(action)

        not_empty_frames = [f for f in frames if f[0] != 0]

        if len(not_empty_frames):
            last_frame = not_empty_frames[-1]

            center_reward = 1 if is_target_at_the_center(last_frame[0:4]) else 0

            reward = reward + center_reward

            return frames, reward, terminated, info

        return frames, reward, terminated, info
