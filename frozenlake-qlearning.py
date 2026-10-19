import gymnasium as gym
import numpy as np
import base64
import glob
import os

LEARNING_RATE = 0.1       # alpha
DISCOUNT_FACTOR = 0.99    # gamma

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9999

NUM_EPISODES = 50000
MAX_STEPS = 500


env = gym.make(
    'FrozenLake-v1',
    desc=None,
    map_name="8x8",
    is_slippery=True
)

num_states = env.observation_space.n
num_actions = env.action_space.n

Q = np.zeros((num_states, num_actions))

def choose_action(state, epsilon):
    #explore
    if np.random.random() < epsilon:
        return env.action_space.sample()

    # choose best action
    return int(np.argmax(Q[state]))


epsilon = EPSILON_START
episode_rewards = []

for episode in range(NUM_EPISODES):
    state, info = env.reset()
    total_reward = 0

    for step in range(MAX_STEPS):
        action = choose_action(state, epsilon)

        next_state, reward, terminated, truncated, info = (
            env.step(action)
        )

        # Q-learning target
        if terminated:
            target = reward
        else:
            target = (
                reward
                + DISCOUNT_FACTOR * np.max(Q[next_state])
            )

        # Q-learning update
        Q[state, action] += LEARNING_RATE * (
            target - Q[state, action]
        )

        total_reward += reward
        state = next_state

        if terminated or truncated:
            break

    episode_rewards.append(total_reward)

    epsilon = max(
        EPSILON_MIN,
        epsilon * EPSILON_DECAY
    )

# training eval ---
video_dir = "./frozenlake_recording"
eval_env = gym.make(
    "FrozenLake-v1",
    map_name="8x8",
    is_slippery=True,
    render_mode="rgb_array"
)
eval_env = gym.wrappers.RecordVideo(
    eval_env,
    video_folder=video_dir,
    episode_trigger=lambda episode_id: True,
    name_prefix="eval",
)
NUM_EVAL_EPISODES = 1
successes = 0

for episode in range(NUM_EVAL_EPISODES):
    state, info = eval_env.reset()

    for step in range(MAX_STEPS):
        best_actions = np.flatnonzero(
            Q[state] == np.max(Q[state])
        )
        action = int(np.random.choice(best_actions))

        state, reward, terminated, truncated, info = (
            eval_env.step(action)
        )

        if terminated or truncated:
            successes += int(reward == 1)
            break

eval_env.close()

print("\nTraining complete!")
print(f"Final epsilon: {epsilon:.4f}")
print(f"Evaluation success rate: {successes / NUM_EVAL_EPISODES:.2%}")
