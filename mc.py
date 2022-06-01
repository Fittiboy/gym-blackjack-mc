from collections import defaultdict

import pandas as pd
import gym


def get_optimal_policy():
    env = gym.make("Blackjack-v0")
    actions = list(range(env.action_space.n))
    policy = defaultdict(env.action_space.sample)
    q_values = defaultdict(float)
    num_iterations = 30
    num_episodes = 2000000
    epsilon = 0.5

    def behavior_policy(state):
        return env.action_space.sample()

    def generate_episode():
        episode = []
        state = env.reset()
        while True:
            action = behavior_policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        return episode

    def evaluate_policy(policy):
        weight_sums = defaultdict(lambda: 0.)
        for _ in range(num_episodes):
            episode = generate_episode()
            returns = 0
            weight = 1
            for step in reversed(episode):
                state, action, reward = step
                returns += reward
                weight_sums[state, action] += weight
                try:
                    q_values[state, action] += \
                        (weight / weight_sums[state, action]) * \
                        (returns - q_values[state, action])
                except ZeroDivisionError:
                    pass
                best_action = max(actions, key=lambda x: q_values[state, x])
                if policy[state] != action:
                    policy[state] = best_action
                    continue
                weight /= epsilon
        return policy

    for iteration in range(num_iterations):
        policy = evaluate_policy(policy)

    return policy


if __name__ == "__main__":
    policy = get_optimal_policy()
    policy = pd.DataFrame(policy.items(), columns=['state', 'action'])
    policy.to_csv("./policy.csv", sep="\t")
