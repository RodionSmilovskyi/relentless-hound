import os
import datetime
import gym
import tensorflow as tf
from gym.wrappers import RecordVideo
from package import settings

EPISODES = 1

settings.OUTPUT_PATH = os.path.join(settings.WORKING_DIRECTORY, "output")

if __name__ == "__main__":
    pendulum = gym.make('Pendulum-v1', g=9.81)
    env = RecordVideo(
        pendulum,
        f"{settings.OUTPUT_PATH}/data/videos",
        episode_trigger=lambda x: True,
        name_prefix=datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
    )

    # interpreter = tf.lite.Interpreter(
    #     os.path.join(settings.WORKING_DIRECTORY, "model", "policy.tflite")
    # )
    # policy_runner = interpreter.get_signature_runner()

    for i in range(EPISODES):
        state = env.reset()

        terminated: bool = False

        while not terminated:

            # inference = policy_runner(
            #     **{
            #         "0/discount": tf.constant(0.0),
            #         "0/observation": tf.cast(tf.constant(state), tf.float32),
            #         "0/reward": tf.constant(0.0),
            #         "0/step_type": tf.constant(0),
            #     }
            # )

            # new_state, reward, terminated, info = env.step(inference['action'][0])

            action = env.action_space.sample()
            new_state, reward, terminated, info = env.step(action)
            print(f"reward {reward}")

            state = new_state

    env.close()
