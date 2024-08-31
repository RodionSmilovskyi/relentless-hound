import os
import datetime
import numpy as np
import tensorflow as tf
from gym.wrappers import FrameStack, RecordVideo
from package import settings
from package.envs.cv_wrapper import CvWrapper
from package.envs.drone import DroneEnv
from package.envs.reward_wrapper import RewardWrapper

EPISODES = 1

settings.OUTPUT_PATH = os.path.join(settings.WORKING_DIRECTORY, "output")

if __name__ == "__main__":
    env = RecordVideo(
        RewardWrapper(FrameStack(CvWrapper(DroneEnv(True)), 4)),
        f"{settings.OUTPUT_PATH}/data/videos",
        episode_trigger=lambda x: True,
        name_prefix=datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
    )

    interpreter = tf.lite.Interpreter(
        os.path.join(settings.WORKING_DIRECTORY, "model", "policy.tflite")
    )
    policy_runner = interpreter.get_signature_runner()

    for i in range(EPISODES):
        state = env.reset()

        terminated: bool = False

        while not terminated:

            inference = policy_runner(
                **{
                    "0/discount": tf.constant(0.0),
                    "0/observation": tf.cast(tf.constant(state), tf.float32),
                    "0/reward": tf.constant(0.0),
                    "0/step_type": tf.constant(0),
                }
            )

            new_state, reward, terminated, info = env.step(inference['action'][0])

            # action = np.array([10, 0, 20, 10])
            # action = env.action_space.sample()
            # new_state, reward, terminated, info = env.step(action)
            print(f"Reward {reward}")

            state = new_state

    env.close()
