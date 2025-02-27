import numpy as np
import matplotlib.pyplot as plt


"""
*
"""
class GeometricBrownianMotion:
    """
    *
    """
    def __init__(self, initial_value, time_period, time_step, mu, sigma):
        self.initial_value = initial_value
        self.time_period = time_period
        self.time_step = time_step
        self.mu = mu
        self.sigma = sigma
    
    """
    *
    """
    def simulate(self, initial_value=0, time_period=0, time_step=0, mu=0, sigma=0):
        return self._geometric_brownian_motion(
            initial_value=initial_value or self.initial_value,
            time_period=time_period or self.time_period,
            time_step=time_step or self.time_step,
            mu=mu or self.mu,
            sigma=sigma or self.sigma,
        )
    
    """
    *
    """
    def simulate_many(self, num_simulations, initial_value=0, time_period=0, time_step=0, mu=0, sigma=0):
        simulations = []
        
        for _ in range(num_simulations):
            simulations.append(self._geometric_brownian_motion(
                initial_value=initial_value or self.initial_value,
                time_period=time_period or self.time_period,
                time_step=time_step or self.time_step,
                mu=mu or self.mu,
                sigma=sigma or self.sigma,
            ))

        return np.vstack(simulations)

    """
    *
    """
    def _geometric_brownian_motion(self, initial_value, time_period, time_step, mu, sigma):
        # Number of time steps
        n_steps = int(time_period / time_step)
        
        # Create an array to store the values at each time step
        prices = np.zeros(n_steps)
        prices[0] = initial_value
        
        # Simulate the GBM process
        for t in range(1, n_steps):
            dt = time_step
            # Calculate the random component (Brownian motion)
            dz = np.random.normal(0, 1) * np.sqrt(dt)
            # Update the price based on the GBM formula
            prices[t] = prices[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dz)
        
        return prices


####################################
######### EXAMPLE USAGE ############
####################################

num_simulations = 5

initial_value = 100
time_period = 2
time_step = time_period / 252
mu = 0.1
sigma = 0.25

gbm = GeometricBrownianMotion(initial_value, time_period, time_step, mu, sigma)

gbms = gbm.simulate_many(num_simulations)

time_array = np.arange(0, time_period, time_step)

for i in range(len(gbms)):
    plt.plot(time_array, gbms[i], label=f"GBM Path {i}")

plt.title("Geometric Brownian Motion Simulation")
plt.xlabel("Time (Years)")
plt.ylabel("Price")
plt.show()
