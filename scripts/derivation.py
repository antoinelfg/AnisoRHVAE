import numpy as np
import scipy.stats as stats
import math

target = {
    'r_0': 1.2,
    'kappa': 7.276,
    'gamma': 7.624,
    's': 8.871,
    'p': 1.744,
    'beta': 9.252,
    'p_atom': 1.096
}

tau = 0.5
d = 2
sigma = tau / np.sqrt(2)
var = tau**2 / 2

print(f"tau={tau}, d={d}, sigma={sigma:.4f}, var={var:.4f}")
print("--- r_0 ---")
conf_997 = stats.chi2.ppf(0.9973, df=d)  # ~3.4 sigma equivalent
r_0_theory_3sigma = sigma * np.sqrt(conf_997)
print(f"1D 3-sigma equivalent for 2D (99.73%): r_0 = {r_0_theory_3sigma:.4f} vs target {target['r_0']}")
print(f"Mahalanobis ^2 = {target['r_0']**2 / var:.4f}")

print("\n--- gamma ---")
print(f"1/var = {1/var:.2f} (theoretical precision for posterior)")
print(f"target gamma = {target['gamma']}")

print("\n--- kappa ---")
k_var_match = np.pi / (tau * np.sqrt(3)) * np.sqrt(2) # pi/tau * sqrt(2/3)
print(f"Variance matching kappa = {k_var_match:.4f} vs target {target['kappa']}")
# What if kappa matches log-gradient
# max gradient of Gaussian Mahalanobis is 2*r_0 / tau^2
k_grad_match = 2 * r_0_theory_3sigma / (tau**2)
print(f"Gradient matching kappa = 2*r_0/tau^2 = {k_grad_match:.4f}")
# What if kappa = 4 * r_0 / tau^2 ? 
print(f"4*r_0/tau^2 = {4 * target['r_0'] / tau**2:.4f}")
print(f"Kappa * tau = {target['kappa'] * tau:.4f}")

print("\n--- s and p ---")
print(f"p = {target['p']}, d - 1/4 = {d - 0.25}")
print(f"s = {target['s']}")
print(f"s / r_0 = {target['s']/target['r_0']:.4f}")
print(f"e^2 / var = {np.exp(2)/var:.4f}")
print(f"s / (2*pi*r_0) = {target['s'] / (2 * np.pi * target['r_0']):.4f}")

print("\n--- beta and p_atom ---")
print(f"beta = {target['beta']}")
print(f"p_atom = {target['p_atom']}")
print(f"d / 2 = {d/2}")
