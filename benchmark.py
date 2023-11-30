from epsilon_greedy import *
from ucb1 import *
from gpucb4 import *
from linucb11 import *




# Parameter settings
# K: Number of arms
# T: Number of time steps
# D: Dimension of each arm
# N: Number of trials

K = 10
T = 12000
D = 10
N = 10

L = 0.01
E = 0.1

#NOTE Good setting: 10 arms, 1000 time steps, 10 dimensions, 100 trials, lambda = 0.005
# for mean w of 10

ucb1_reward, ucb1_regret, ucb1_uncertainty = ucb1_trials(K, T, D, N)
eg_reward, eg_regret, eg_uncertainty = epsilon_greedy_trials(K, T, D, N, E)
# gpucb_reward, gpucb_regret, gpucb_uncertainty = gpucb_trials(K, T, D, N)
linucb_reward, linucb_regret, linucb_uncertainty, linucb_error = linucb_trials(K, T, D, N, L)



fig, axs = plt.subplots(2, 2, figsize=(15, 7.5))

X = list(range(1, T + 1))

axs[0, 0].plot(X, ucb1_reward, color='blue', label='UCB1')
axs[0, 0].plot(X, eg_reward, color='red', label='Epsilon Greedy')
# axs[0, 0].plot(X, gpucb_reward)
axs[0, 0].plot(X, linucb_reward, color='green', label='LinUCB')
axs[0, 0].set_xlabel('Time Step')
axs[0, 0].set_ylabel('Cumulative Reward')
axs[0, 0].set_title('Average Cumulative Reward (' + str(N) + ' Trials)')

axs[0, 1].plot(X, ucb1_regret, color='blue', label='UCB1')
axs[0, 1].plot(X, eg_regret, color='red', label='Epsilon Greedy')
# axs[0, 1].plot(X, gpucb_regret)
axs[0, 1].plot(X, linucb_regret, color='green', label='LinUCB')
axs[0, 1].set_xlabel('Time Step')
axs[0, 1].set_ylabel('Cumulative Regret')
axs[0, 1].set_title('Average Cumulative Regret (' + str(N) + ' Trials)')

axs[1, 0].plot(X, ucb1_uncertainty, color='blue', label='UCB1')
axs[1, 0].plot(X, eg_uncertainty, color='red', label='Epsilon Greedy')
# axs[1, 0].plot(X, gpucb_uncertainty)
axs[1, 0].plot(X, linucb_uncertainty, color='green', label='LinUCB')
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