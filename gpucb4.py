import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


# Where should sigma be defined in this algorithm?
def rbf(x, y, sigma):
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))


def gram_matrix(X1, X2, sigma):
    gram = np.zeros((len(X1), len(X2)))
    for i in range(len(X1)):
        for j in range(len(X2)):
            gram[i, j] = rbf(X1[i], X2[j], sigma)
    return gram


# Need clarity on sigmas in this section - standard deviation?
# X: Training data stacked
# Y: Training data labels
# X_: New data point
def gp_update(X, Y, X_, sigma, sigma_n):
    mean = gram_matrix(X_, X, sigma) @ np.linalg.inv(gram_matrix(X, X, sigma) + sigma_n ** 2 * np.eye(len(X))) @ Y
    cov = gram_matrix(X_, X_, sigma) - gram_matrix(X_, X, sigma) @ np.linalg.inv(gram_matrix(X, X, sigma) + sigma_n ** 2 * np.eye(len(X))) @ gram_matrix(X, X_, sigma)
    return mean, cov


def gp_ucb(K, t, x, gp):

    # Get argmax arm from GP
    # Examine the reward f(a_t)
    # Update GP with new data point
    # Repeat
    # Plot the GP uncertainty intervals
    # Plot the mean - should be an easy way of doing it

    max_value = 0.0
    a_t = 0

    for a in range(0, K):

        # Reward from GP
        a_array = np.array(a).reshape(-1, 1)
        # print(a_array)
        mean, std = gp.predict(a_array, return_std=True)
        # print("Std: ", std)
        # print("Std ** 2: ", std ** 2)
        r_a_t = (mean + std * np.sqrt(np.log(t+1)))[0]

        # print("Arm " + str(a) + " mean: " + str(mean))
        # print("Arm " + str(a) + " UCB: " + str(r_a_t))

        if r_a_t > max_value: 
            max_value = r_a_t
            a_t = a

    return a_t


def gpucb_trials(K, T, D, N):

    # Arrays for tracking cumulative statistics
    average_reward              = np.zeros(T)
    average_regret              = np.zeros(T)
    average_uncertainty         = np.zeros(T)
    average_estimation_error    = np.zeros(T)

    progress_bar = tqdm(total=N, unit="iteration")

    for i in range(N):

        # Random values with a mean of 0.5
        w = np.random.multivariate_normal(np.full(D, 0.5), np.eye(D), size=K)  

        # Random values between 0 and 1
        x = np.random.rand(K, D)  

        # thetas = np.random.multivariate_normal(np.full(D, 10), np.eye(D), size=K)  

        cumulative_rewards  = np.zeros(T)
        cumulative_regret   = np.zeros(T)
        regret_array        = np.zeros(T)

        uncertainties       = np.zeros((K, T))
        # Estimation error for the optimal arm
        estimation_error    = np.zeros(T)
        ucb_array           = np.zeros((T, K))

        kernel = RBF(length_scale=100) + WhiteKernel(noise_level=1e-5)
        gp = GaussianProcessRegressor(kernel=kernel)
        X = [[0]]
        y = [0]

        gp.fit(X, y)

        kernel = gp.kernel(X, X)  
        max_K = np.max(np.abs(kernel))

        while max_K > 1:
            gp.kernel_.length_scale *= 0.95 
            gp.fit(X, y)  
            kernel = gp.kernel_.k1 + gp.kernel_.k2
            max_K = np.max(np.abs(kernel))


        # Need more configuration for GP?
        # gp = GaussianProcessRegressor(kernel=RBF(), alpha=0.1)
        # init_X = np.array([0]).reshape(-1, 1)
        # init_Y = np.array([0])
        # gp.fit(init_X, init_Y)

        for t in range(0, T):

            # print("Time step " + str(t))

            a_t_max = gp_ucb(K, t, X, gp)
            reward = np.dot(w[a_t_max], x[a_t_max])

            # print("Arm " + str(a_t_max) + " reward: " + str(reward))

            X = np.vstack((X, np.array(a_t_max).reshape(-1, 1)))
            y = np.vstack((y, reward))
            gp.fit(X, y)

            mean, cov = gp_update(X, y, np.array(a_t_max).reshape(-1, 1), 0.1, 0.1)
            # print("Mean: ", mean)
            # print("Cov: ", cov)

            # How should noise scale?
            # reward_array = [(np.dot(w[i], x[i]) * (1 + random.gauss(0, 0.01))) for i in range(K)]
            # Temporarily remove noise
            reward_array = [np.dot(w[i], x[i]) for i in range(K)]
            # reward = reward_array[a_t_max]
            optimal_arm = np.argmax(reward_array)

            cumulative_rewards[t] = cumulative_rewards[t-1] + reward

            # estimation_error[t] = np.linalg.norm(w[optimal_arm] - thetas[optimal_arm])
            # if t == 0:
            #     ee_normalization = estimation_error[t]

            regret = reward_array[optimal_arm] - reward

            regret_array[t] = regret
            cumulative_regret[t] = cumulative_regret[t-1] + regret

        # Update cumulative statistics
        average_reward              = np.add(average_reward, cumulative_rewards)
        average_regret              = np.add(average_regret, cumulative_regret)
        average_uncertainty         = np.add(average_uncertainty, uncertainties[optimal_arm])
        # average_estimation_error    = np.add(average_estimation_error, estimation_error / ee_normalization)

        progress_bar.update(1)
    
    progress_bar.close()
    return average_reward / N, average_regret / N, average_uncertainty / N



if __name__ == "__main__":

    # Parameter settings
    # K: Number of arms
    # T: Number of time steps
    # D: Dimension of each arm
    # N: Number of trials
    # L: Lambda value for covariance matrix initialization

    K = 10
    T = 40
    D = 3
    N = 1
    L = 0.1

    #NOTE Good setting: 10 arms, 1000 time steps, 10 dimensions, 100 trials, lambda = 0.005
    # for mean w of 10

    average_reward, average_regret, average_uncertainty = gpucb_trials(K, T, D, N)

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
    plt.show()