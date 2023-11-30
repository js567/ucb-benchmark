import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


def lin_ucb(K, t, x, A, thetas, uncertainties, ucb_array, D):

    max_value = 0.0
        
    for a in list(range(0, K)):

        uncertainty         = np.sqrt(np.dot(x[a], np.dot(np.linalg.inv(A[a]), x[a])))
        uncertainties[a][t] = uncertainty

        C = (np.sqrt(np.log(1 + (t / D))))
        ucb = np.dot(thetas[a], x[a]) + (C * uncertainty)
        ucb_array[t][a] = ucb

        # Find the arm with the highest UCB
        if ucb >= max_value: 
            max_value = ucb
            a_t = a 

    return a_t


def linucb_trials(K, T, D, N, lambda_):

    # Arrays for tracking cumulative statistics
    average_reward              = np.zeros(T)
    average_regret              = np.zeros(T)
    average_uncertainty         = np.zeros(T)
    average_estimation_error    = np.zeros(T)

    progress_bar = tqdm(total=N, unit="iteration")

    for i in range(N):

        # Random values with a mean of 0.5
        w = np.random.multivariate_normal(np.full(D, 0.5), np.eye(D), size=K)  

        max_norms = np.max(np.linalg.norm(w, axis=1))
        normalized_w = w / max_norms
        w = normalized_w

        # norms = np.linalg.norm(w, axis=1)
        # normalized_w = w / norms[:, np.newaxis]
        # w = normalized_w

        # print("W: ",  w)
        # print(max_norms)

        # Random values between 0 and 1
        x = np.random.rand(K, D)  

        max_norms = np.max(np.linalg.norm(x, axis=1))
        normalized_x = x / max_norms
        x = normalized_x

        # norms = np.linalg.norm(x, axis=1)
        # normalized_x = x / norms[:, np.newaxis]
        # x = normalized_x

        A       = [lambda_ * np.eye(D) for _ in range(K)] 
        A_inv   = [np.linalg.inv(A[a]) for a in range(K)]
        b       = [np.zeros(D) for _ in range(K)]
        thetas  = [np.zeros(D) for _ in range(K)]
        # thetas = np.random.multivariate_normal(np.zeros(D), np.eye(D), size=K)  

        cumulative_rewards  = np.zeros(T)
        cumulative_regret   = np.zeros(T)
        regret_array        = np.zeros(T)

        uncertainties       = np.zeros((K, T))
        # Estimation error for the optimal arm
        estimation_error    = np.zeros(T)
        ucb_array           = np.zeros((T, K))

        for t in range(0, T):

            a_t_max = lin_ucb(K, t, x, A, thetas, uncertainties, ucb_array, D) 

            # How should noise scale?
            # reward_array = [(np.dot(w[i], x[i]) * (1 + random.gauss(0, 0.09))) for i in range(K)]
            # Temporarily remove noise
            reward_array = [np.dot(w[i], x[i]) for i in range(K)]
            reward = reward_array[a_t_max]
            optimal_arm = np.argmax(reward_array)

            # print(np.dot(w[optimal_arm], x[optimal_arm]))
            # print(np.dot(thetas[optimal_arm], x[optimal_arm]))

            cumulative_rewards[t] = cumulative_rewards[t-1] + reward

            estimation_error[t] = np.linalg.norm(w[optimal_arm] - thetas[optimal_arm])
            if t == 0:
                ee_normalization = estimation_error[t]

            regret = reward_array[optimal_arm] - reward

            regret_array[t] = regret
            cumulative_regret[t] = cumulative_regret[t-1] + regret
            
            # Reference repo code
            # Normalize sampled w and x - L2 norm upper bounded by 1
            # Divide x / x_max
            # LinUCB.py on Github

            A[a_t_max]      += np.outer(x[a_t_max], x[a_t_max])
            b[a_t_max]      += reward * x[a_t_max]
            A_inv[a_t_max]  = np.linalg.inv(A[a_t_max])
            thetas[a_t_max] = np.dot(A_inv[a_t_max], b[a_t_max])

            # Questions for Dr. Wang
            # Is there an identifiability issue - yes
            # Is it ok to converge to an alternate theta? - no
            # Regularization should help not approach the wrong theta


        # Update cumulative statistics
        average_reward              = np.add(average_reward, cumulative_rewards)
        average_regret              = np.add(average_regret, cumulative_regret)
        average_uncertainty         = np.add(average_uncertainty, uncertainties[optimal_arm])
        average_estimation_error    = np.add(average_estimation_error, estimation_error / ee_normalization)

        progress_bar.update(1)
    
    progress_bar.close()
    return average_reward / N, average_regret / N, average_uncertainty / N, average_estimation_error / N



if __name__ == "__main__":

    # Parameter settings
    # K: Number of arms
    # T: Number of time steps
    # D: Dimension of each arm
    # N: Number of trials
    # L: Lambda value for covariance matrix initialization

    K = 10
    T = 10000
    D = 10
    N = 2
    L = 0.001

    #NOTE Good setting: 10 arms, 1000 time steps, 10 dimensions, 100 trials, lambda = 0.005
    # for mean w of 10

    average_reward, average_regret, average_uncertainty, average_estimation_error = linucb_trials(K, T, D, N, L)

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

    axs[1, 1].plot(X, average_estimation_error)
    axs[1, 1].set_xlabel('Time Step')
    axs[1, 1].set_ylabel('Estimation Error (L2 Norm)')
    axs[1, 1].set_title('Normalized Average Estimation Error (' + str(N) + ' Trials)')
    axs[1, 1].set_ylim(0, 1)

    fig.subplots_adjust(left=0.15, bottom=0.05, right=0.85, top=0.95)
    # plt.tight_layout()
    plt.show()