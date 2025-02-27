from GeometricBrownianMotion import GeometricBrownianMotion

import numpy as np
import matplotlib.pyplot as plt

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
