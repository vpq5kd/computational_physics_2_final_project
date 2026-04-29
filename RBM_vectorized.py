import numpy as np
import matplotlib.pyplot as plt


class RBM:

    epsilon_w_arr = np.array([])

    def __init__(self, N, M):
        self.N = N
        self.M = M
        self.rng = np.random.default_rng()
        
        self.W = 0.01 * self.rng.standard_normal((N, M))
        self.theta_v = np.zeros(N)
        self.theta_h = np.zeros(M)

    def sample_hidden(self, sigma):
        activation_value = sigma @ self.W + self.theta_h
        prob = 1.0 / (1.0 + np.exp(-activation_value))
        tau = (self.rng.random(prob.shape) < prob).astype(float)
        
        return tau
    
    def sample_visible(self, tau):
        mean = tau @ self.W.T + self.theta_v
        sigma = self.rng.normal(loc=mean, scale=1.0)

        return sigma
    
    def run_cd_k(self, data_set, k):
        sigma = data_set.copy()
        tau = self.sample_hidden(sigma)

        for _ in range(k):
            sigma = self.sample_visible(tau)
            tau = self.sample_hidden(sigma)

        return sigma, tau

    def update_params(self, eta, k, data_set):
        tau_data = self.sample_hidden(data_set)

        sigma_model, tau_model = self.run_cd_k(data_set, k)

        self.W += eta * (
            (data_set.T @ tau_data) / len(data_set)
            - (sigma_model.T @ tau_model) / len(data_set)
        )

        self.theta_v += eta * (
            data_set.mean(axis=0)
            - sigma_model.mean(axis=0)
        )

        self.theta_h += eta * (
            tau_data.mean(axis=0)
            - tau_model.mean(axis=0)
        )
    
    def calculate_epsilon_W(self, w_before):
        epsilon_w = np.mean(np.abs(self.W - w_before))
        return epsilon_w

    def train_model(self, eta, k, data_set, n_epochs):
        epsilon_w_arr = []

        for epoch in range(n_epochs):
            w_before = self.W.copy()
            self.update_params(eta, k, data_set)
            epsilon_w = self.calculate_epsilon_W(w_before)
            epsilon_w_arr.append(epsilon_w)
            print(f"Epoch: {epoch} | epsilon_w: {epsilon_w}")
        
        self.epsilon_w_arr = np.array(epsilon_w_arr)
        return self.epsilon_w_arr

    def save_model(self, filename):
        np.savez(
            filename,
            W=self.W,
            epsilon_w_arr=self.epsilon_w_arr,
            theta_v=self.theta_v,
            theta_h=self.theta_h
        )
    
    def load_model(self, filename):
        data = np.load(filename)
        self.W = data["W"]
        self.theta_v = data["theta_v"]
        self.theta_h = data["theta_h"]
        self.epsilon_w_arr = data["epsilon_w_arr"]

    def visible_mean(self, tau):
        return tau @ self.W.T + self.theta_v

    def hidden_prob(self, sigma):
        activation = sigma @ self.W + self.theta_h
        prob = 1.0/(1.0 + np.exp(-activation))
        return prob


    def generate_rbm_states(self, num_states=1000, melting_iterations=50000):
        states = []
        sigma = self.rng.normal(0.0, 1.0, size=self.N)

        for _ in range(melting_iterations):
            tau = self.sample_hidden(sigma)
            sigma = self.sample_visible(tau)

        for _ in range(num_states):
            tau = self.sample_hidden(sigma)
            sigma = self.sample_visible(tau)
            states.append(sigma.copy())

        return np.array(states)

    def display_epsilon_w(self, filename):
        
        epoch_array = np.arange(len(self.epsilon_w_arr))
        
        plt.figure()
        plt.plot(
            epoch_array,
            self.epsilon_w_arr,
            color='saddlebrown',
            marker='+',
            linestyle='None'
        )
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.xlabel('Epoch Number')
        plt.ylabel(
            r'$\epsilon_W = \frac{1}{NM} \sum_{i, j} |\Delta W_{ij}|$'
        )
        plt.savefig(filename)
        plt.show()
    
    def display_inter_layer_couplings(self, filename):
        
        weights = self.W.flatten()

        plt.figure()
        plt.hist(
            weights,
            bins=30,
            color='palevioletred',
            histtype='stepfilled'
        )
        plt.xlabel(r"$W_{ij}$")
        plt.ylabel("Count")
        plt.savefig(filename)
        plt.show()
