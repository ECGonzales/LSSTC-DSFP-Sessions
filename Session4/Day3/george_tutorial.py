import george
# To check the version : george.__version__ (want version '0.3.0')
import numpy as np
import matplotlib.pyplot as plt
from george import kernels

# We’ll start by generating some fake data (from a sinusoidal model) with error bars:
np.random.seed(1234)  # generate random data
x = 10 * np.sort(np.random.rand(15))  # sort the data
yerr = 0.2 * np.ones_like(x)  # White noise
y = np.sin(x) + yerr * np.random.randn(len(x))

# Create a plot of the data we just generated
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.xlim(0, 10)
plt.ylim(-1.45, 1.45)
plt.xlabel("x")
plt.ylabel("y")

# Now, we’ll choose a kernel (covariance - how the data are related to each other) function to model these data,
# assume a zero mean model, and predict the function values across the full range. The full kernel specification
# language is documented here (see link in .ipynb) but here’s an example for this dataset:

kernel = np.var(y) * kernels.ExpSquaredKernel(0.5)  # Define function that computes the variance along y using np.var
gp = george.GP(kernel)                              # and multiplies it by the kernel with the metric being 0.5 (length scale)
gp.compute(x, yerr)

x_pred = np.linspace(0, 10, 500)
pred, pred_var = gp.predict(y, x_pred, return_var=True)  # computes the predictions, return the variance

plt.fill_between(x_pred, pred - np.sqrt(pred_var), pred + np.sqrt(pred_var),  # plot the uncertainties
                color="k", alpha=0.2)
plt.plot(x_pred, pred, "k", lw=1.5, alpha=0.5)  # plot the gp mean (fit)
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.plot(x_pred, np.sin(x_pred), "--g")  # plot the x prediction as a sin function
plt.xlim(0, 10)
plt.ylim(-1.45, 1.45)
plt.xlabel("x")
plt.ylabel("y")

print("Initial ln-likelihood: {0:.2f}".format(gp.log_likelihood(y)))

# So we can use this—combined with scipy’s minimize function—to fit for the maximum likelihood parameters:
from scipy.optimize import minimize

def neg_ln_like(p):  # Get negative log likelihood
    gp.set_parameter_vector(p)
    return -gp.log_likelihood(y)

def grad_neg_ln_like(p):  # get the gradient of ln likelihood so we can then find the minimum
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(y)

result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)  # Get that minimum
print(result)

gp.set_parameter_vector(result.x)
print("\nFinal ln-likelihood: {0:.2f}".format(gp.log_likelihood(y)))
# the last line shows the [Amplitude, length(?)

# And plot the maximum likelihood model:
pred, pred_var = gp.predict(y, x_pred, return_var=True)  # give me prediction and variance

plt.fill_between(x_pred, pred - np.sqrt(pred_var), pred + np.sqrt(pred_var),
                color="k", alpha=0.2)
plt.plot(x_pred, pred, "k", lw=1.5, alpha=0.5)
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.plot(x_pred, np.sin(x_pred), "--g")
plt.xlim(0, 10)
plt.ylim(-1.45, 1.45)
plt.xlabel("x")
plt.ylabel("y")
