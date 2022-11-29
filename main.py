import numpy as np

from gym_vrp.envs import TSPEnv

# Environment
env = TSPEnv(
    num_nodes=10,
    batch_size=1,
    num_draw=1,
    seed=69
)


if __name__ == '__main__':
    env.enable_video_capturing("videos/test1.mp4")
    for i in range(10):
        action = np.array([[i]])
        state, reward, done, _ = env.step(action)
        print(state[:,:,3])