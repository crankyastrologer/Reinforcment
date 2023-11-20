import random
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.core.pylabtools import figsize


class PowerSocket:
    """ the base power socket class """

    def __init__(self, q):
        self.q = q  # the true reward value
        self.initialize()  # reset the socket

    def initialize(self):
        self.Q = 0  # the estimate of this socket's reward value
        self.n = 0  # the number of times this socket has been tried

    def charge(self):
        """ return a random amount of charge """

        # the reward is a guassian distribution with unit variance around the true
        # value 'q'
        value = np.random.randn() + self.q

        # never allow a charge less than 0 to be returned
        return 0 if value < 0 else value

    def update(self, R):
        """ update this socket after it has returned reward value 'R' """

        # increment the number of times this socket has been tried
        self.n += 1

        # the new estimate of the mean is calculated from the old estimate
        self.Q = (1 - 1.0 / self.n) * self.Q + (1.0 / self.n) * R

    def sample(self, t):
        """ return an estimate of the socket's reward value """
        return self.Q


# create 5 sockets in a fixed order
class OptimisticSocket(PowerSocket):
    def __init__(self, q, **kwargs):
        # get the initial estimate from the kwargs
        self.initial_estimate = kwargs.pop('initial_estimate', 0.)

        # pass the true reward value to the base PowerSocket
        super().__init__(q)

    def initialize(self):
        # estimate of this socket's reward value
        # - set to supplied initial value
        self.Q = self.initial_estimate

        # the number of times this socket has been tried
        # - set to 1 if an initialisation value is supplied
        self.n = 1 if self.initial_estimate > 0 else 0
socket_order = [3,5,1,2,4]

# create the sockets
# - the mean value of each socket is derived from the socket order index, which is doubled to give
#   distinct values and offset by 2 to keep the distribution above zero
sockets = [PowerSocket((q * 2) + 2) for q in socket_order]

# save the number of sockets
NUM_SOCKETS = len(socket_order)
# using a very large number of time steps just to create sufficient data to get smooth plots of socket output
TOTAL_STEPS = 1000

# rewards will contain the charge returned at all of the time steps for each socket
rewards = np.zeros(shape=(TOTAL_STEPS, NUM_SOCKETS))

# iterate through each of the sockets
for socket_number, socket in enumerate(sockets):

    # get charge from the socket for the defined number of steps
    for t in range(TOTAL_STEPS): rewards[t, socket_number] = socket.charge()

# plot the charge we got from the sockets

reward = pd.DataFrame(rewards, columns = socket_order)
print(reward.sum())
plt.violinplot(rewards)
plt.xlabel('Socket')
plt.ylabel('Reward Distribution (seconds of charge)')
plt.title('Violin plot of the reward distribution for each socket')
plt.show()


class SocketTester():
    """ create and test a set of sockets over a single test run """

    def __init__(self, socket=PowerSocket, socket_order=socket_order, multiplier=2, **kwargs):

        # create supplied socket type with a mean value defined by socket order
        self.sockets = [socket((q * multiplier) + 2, **kwargs) for q in socket_order]

        # set the number of sockets equal to the number created
        self.number_of_sockets = len(self.sockets)

        # the index of the best socket is the last in the socket_order list
        # - this is a one-based value so convert to zero-based
        self.optimal_socket_index = (socket_order[-1] - 1)

        # by default a socket tester records 2 bits of information over a run
        self.number_of_stats = kwargs.pop('number_of_stats', 2)

    def initialize_run(self, number_of_steps):
        """ reset counters at the start of a run """

        # save the number of steps over which the run will take place
        self.number_of_steps = number_of_steps

        # reset the actual number of steps that the test ran for
        self.total_steps = 0

        # monitor the total reward obtained over the run
        self.total_reward = 0

        # the current total reward at each timestep of the run
        self.total_reward_per_timestep = []

        # the actual reward obtained at each timestep
        self.reward_per_timestep = []

        # stats for each time-step
        # - by default records: estimate, number of trials
        self.socket_stats = np.zeros(shape=(number_of_steps + 1,
                                            self.number_of_sockets,
                                            self.number_of_stats))

        # ensure that all sockets are re-initialized
        for socket in self.sockets: socket.initialize()

    def charge_and_update(self, socket_index):
        """ charge from & update the specified socket and associated parameters """

        # charge from the chosen socket and update its mean reward value
        reward = self.sockets[socket_index].charge()
        self.sockets[socket_index].update(reward)

        # update the total reward
        self.total_reward += reward

        # store the current total reward at this timestep
        self.total_reward_per_timestep.append(self.total_reward)

        # store the reward obtained at this timestep
        self.reward_per_timestep.append(reward)

    def get_socket_stats(self, t):
        """ get the current information from each socket """
        socket_stats = [[socket.Q, socket.n] for socket in self.sockets]
        return socket_stats

    def get_mean_reward(self):
        """ the total reward averaged over the number of time steps """
        return (self.total_reward / self.total_steps)

    def get_total_reward_per_timestep(self):
        """ the cumulative total reward at each timestep of the run """
        return self.total_reward_per_timestep

    def get_reward_per_timestep(self):
        """ the actual reward obtained at each timestep of the run """
        return self.reward_per_timestep

    def get_estimates(self):
        """ get the estimate of each socket's reward at each timestep of the run """
        return self.socket_stats[:, :, 0]

    def get_number_of_trials(self):
        """ get the number of trials of each socket at each timestep of the run """
        return self.socket_stats[:, :, 1]

    def get_socket_percentages(self):
        """ get the percentage of times each socket was tried over the run """
        return (self.socket_stats[:, :, 1][self.total_steps] / self.total_steps)

    def get_optimal_socket_percentage(self):
        """ get the percentage of times the optimal socket was tried """
        final_trials = self.socket_stats[:, :, 1][self.total_steps]
        return (final_trials[self.optimal_socket_index] / self.total_steps)

    def get_time_steps(self):
        """ get the number of time steps that the test ran for """
        return self.total_steps

    def select_socket(self, t):
        """ Greedy Socket Selection"""

        # choose the socket with the current highest mean reward or arbitrarily
        # select a socket in the case of a tie
        socket_index = random_argmax([socket.sample(t + 1) for socket in self.sockets])
        return socket_index

    def run(self, number_of_steps, maximum_total_reward=float('inf')):
        """ perform a single run, over the set of sockets,
            for the defined number of steps """

        # reset the run counters
        self.initialize_run(number_of_steps)

        # loop for the specified number of time-steps
        for t in range(number_of_steps):

            # get information about all sockets at the start of the time step
            self.socket_stats[t] = self.get_socket_stats(t)

            # select a socket
            socket_index = self.select_socket(t)

            # charge from the chosen socket and update its mean reward value
            self.charge_and_update(socket_index)

            # test if the accumulated total reward is greater than the maximum
            if self.total_reward > maximum_total_reward:
                break

        # save the actual number of steps that have been run
        self.total_steps = t

        # get the stats for each socket at the end of the run
        self.socket_stats[t + 1] = self.get_socket_stats(t + 1)

        return self.total_steps, self.total_reward
def random_argmax(value_list):
    """ a random tie-breaking argmax"""
    values = np.asarray(value_list)
    return np.argmax(np.random.random(values.shape) * (values == values.max()))


beta = stats.beta

params = [(1, 1), (4, 6), (90, 60)]
x = np.linspace(0.0, 1.0, 10000)

plt.figure(figsize=(12, 7))

colors = ["red", "blue", "green"]
c_index = 0

for α, β in params:
    y = beta.pdf(x, α, β)
    c = colors[c_index]
    lines = plt.plot(x, y, label=f"({α},{β})", lw=3, color=c)
    plt.fill_between(x, 0, y, alpha=0.2, color=c)

    if α > 1:
        mean = α / (α + β)
        plt.vlines(mean, 0, beta.pdf(mean, α, β), colors=c, linestyles="--", lw=2)

    plt.autoscale(tight=True)
    c_index += 1


class EpsilonGreedySocketTester(SocketTester):

    def __init__(self, epsilon=0.):

        # create a standard socket tester
        super().__init__()

        # save the probability of selecting the non-greedy action
        self.epsilon = epsilon

    def select_socket(self, t):
        """ Epsilon-Greedy Socket Selection"""

        # probability of selecting a random socket
        p = np.random.random()

        # if the probability is less than epsilon then a random socket is chosen from the complete set
        if p < self.epsilon:
            socket_index = np.random.choice(self.number_of_sockets)
        else:
            # choose the socket with the current highest mean reward or arbitrary select a socket in the case of a tie
            socket_index = random_argmax([socket.sample(t) for socket in self.sockets])

        return socket_index
plt.title('The Beta Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
tester = EpsilonGreedySocketTester( epsilon = 0.2 )
b = tester.run( number_of_steps = 1000 )

plt.ylim(0, 10.2)
plt.legend(loc='upper left', title="(α,β) parameters");
tester = SocketTester( OptimisticSocket, initial_estimate = 20.)
a = tester.run( number_of_steps = 1000 )


print(f' optimistic total steps = {a[0]} totalreward = {a[1]}')
print(f'epsilon greedy total steps = {b[0]} totalreward = {b[1]}')

