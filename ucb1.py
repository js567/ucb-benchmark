import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


def ucb_1(K, t, pull_number, R, ucb_array):

    max_value = 0.0
        
    for a in list(range(0, K)):

        # uncertainty         = np.sqrt(np.dot(x[a], np.dot(np.linalg.inv(A[a]), x[a])))
        # uncertainties[a][t] = uncertainty

        C = np.sqrt(2 * np.log(1 + t)) / (pull_number[a] + 1)
        ucb = R[a] + C 
        ucb_array[t][a] = ucb

        # Find the arm with the highest UCB
        if ucb >= max_value: 
            max_value = ucb
            a_t = a 

    return a_t


def ucb1_trials(K, T, D, N):

    # Arrays for tracking cumulative statistics
    average_reward              = np.zeros(T)
    average_regret              = np.zeros(T)
    average_uncertainty         = np.zeros(T)

    progress_bar = tqdm(total=N, unit="iteration")

    for i in range(N):

        # Random values with a mean of 0.5
        w = np.random.multivariate_normal(np.full(D, 0.5), np.eye(D), size=K)  

        max_norms = np.max(np.linalg.norm(w, axis=1))
        normalized_w = w / max_norms
        w = normalized_w

        # Random values between 0 and 1
        x = np.random.rand(K, D)  

        max_norms = np.max(np.linalg.norm(x, axis=1))
        normalized_x = x / max_norms
        x = normalized_x

        pull_number = np.zeros(K)
        R = np.zeros(K)

        cumulative_rewards  = np.zeros(T)
        cumulative_regret   = np.zeros(T)
        regret_array        = np.zeros(T)

        uncertainties       = np.zeros((K, T))
        ucb_array           = np.zeros((T, K))

        for t in range(0, T):

            a_t_max = ucb_1(K, t, pull_number, R, ucb_array) 

            pull_number[a_t_max] += 1
            R[a_t_max] += np.dot(w[a_t_max], x[a_t_max]) / pull_number[a_t_max]

            # How should noise scale?
            reward_array = [(np.dot(w[i], x[i]) * (1 + random.gauss(0, 0.15))) for i in range(K)]
            # Temporarily remove noise
            # reward_array = [np.dot(w[i], x[i]) for i in range(K)]
            reward = reward_array[a_t_max]
            optimal_arm = np.argmax(reward_array)

            cumulative_rewards[t] = cumulative_rewards[t-1] + reward

            regret = reward_array[optimal_arm] - reward

            regret_array[t] = regret
            cumulative_regret[t] = cumulative_regret[t-1] + regret

        # Update cumulative statistics
        average_reward              = np.add(average_reward, cumulative_rewards)
        average_regret              = np.add(average_regret, cumulative_regret)
        average_uncertainty         = np.add(average_uncertainty, uncertainties[optimal_arm])

        progress_bar.update(1)
    
    progress_bar.close()
    return average_reward / N, average_regret / N, average_uncertainty / N



if __name__ == "__main__":

    # Parameter settings
    # K: Number of arms
    # T: Number of time steps
    # D: Dimension of each arm
    # N: Number of trials

    K = 10
    T = 100
    D = 10
    N = 20

    #NOTE Good setting: 10 arms, 1000 time steps, 10 dimensions, 100 trials, lambda = 0.005
    # for mean w of 10

    average_reward, average_regret, average_uncertainty = ucb1_trials(K, T, D, N)

    # Create plots

    fig, axs = plt.subplots(2, 2, figsize=(15, 7.5))

    X = list(range(1, T + 1))

    axs[0, 0].plot(X, average_reward)
    axs[0, 0].set_xlabel('Time Step')
    axs[0, 0].set_ylabel('Cumulative Reward')
    axs[0, 0].set_title('Average Cumulative Reward (' + str(N) + ' Trials)')

    axs[0, 1].plot(X, average_regret)
    axs[0, 1].set_xlabel('Time Step')
    axs[0, 1].set_ylabel('Cumulative Regret')
    axs[0, 1].set_title('Average Cumulative Regret (' + str(N) + ' Trials)')

    axs[1, 0].plot(X, average_uncertainty)
    axs[1, 0].set_xlabel('Time Step')
    axs[1, 0].set_ylabel('Uncertainty')
    axs[1, 0].set_title('Average Uncertainty (' + str(N) + ' Trials)')

    # axs[1, 1].plot(X, average_estimation_error)
    # axs[1, 1].set_xlabel('Time Step')
    # axs[1, 1].set_ylabel('Estimation Error (L2 Norm)')
    # axs[1, 1].set_title('Normalized Average Estimation Error (' + str(N) + ' Trials)')
    # axs[1, 1].set_ylim(0, 1)

    fig.subplots_adjust(left=0.15, bottom=0.05, right=0.85, top=0.95)
    # plt.tight_layout()
    plt.show()