import george
import numpy as np
import matplotlib.pyplot as pl
from george import kernels

'''In this example, we’re going to simulate a common data analysis situation where our dataset exhibits unknown 
correlations in the noise. When taking data, it is often possible to estimate the independent measurement uncertainty on
 a single point (due to, for example, Poisson counting statistics) but there are often residual systematics that 
 correlate data points. The effect of this correlated noise can often be hard to estimate but ignoring it can introduce 
 substantial biases into your inferences. In the following sections, we will consider a synthetic dataset with 
 correlated noise and a simple non-linear model. We will start by fitting the model assuming that the noise is 
 uncorrelated and then improve on this model by modeling the covariance structure in the data using a Gaussian process.
 '''

# Simple mean model
'''The model that we’ll fit in this demo is a single Gaussian feature with three parameters: amplitude α, location ℓ,
 and width σ2. I’ve chosen this model because is is the simplest non-linear model that I could think of, and it is
  qualitatively similar to a few problems in astronomy (fitting spectral features, measuring transit times, etc.).'''

# ----- Simulate the Dataset ----
# Parameters: alpha = -1, l =0.1, sigma^2= 0.4
from george.modeling import Model

# create a classs for the model we will use
class Model(Model):
    parameter_names = ("amp", "location", "log_sigma2")

    def get_value(self, t):
        return self.amp * np.exp(-0.5*(t.flatten()-self.location)**2 * np.exp(-self.log_sigma2))


# Create the data set we will use
np.random.seed(1234)


def generate_data(params, N, rng=(-5, 5)):
    gp = george.GP(0.1 * kernels.ExpSquaredKernel(3.3))  # create the GP
    t = rng[0] + np.diff(rng) * np.sort(np.random.rand(N))  # x axis: time
    y = gp.sample(t)
    y += Model(**params).get_value(t)
    yerr = 0.05 + 0.05 * np.random.rand(N)
    y += yerr * np.random.randn(N)
    return t, y, yerr

truth = dict(amp=-1.0, location=0.1, log_sigma2=np.log(0.4))  # parameters for model/truth?
t, y, yerr = generate_data(truth, 50)  # create the data

pl.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
pl.ylabel(r"$y$")
pl.xlabel(r"$t$")
pl.xlim(-5, 5)
pl.title("simulated data")

# ---- Adding white noise ------
# Assume white noise is uncorrelated and include a linear trend for systematics in the data
# create the model
class PolynomialModel(Model):
    parameter_names = ("m", "b", "amp", "location", "log_sigma2")

    def get_value(self, t):
        t = t.flatten()
        return (t * self.m + self.b +
                self.amp * np.exp(-0.5*(t-self.location)**2*np.exp(-self.log_sigma2)))


#################### Not done!!!! finish this point on ###########
model = george.GP(mean=PolynomialModel(m=0, b=0, amp=-1, location=0.1, log_sigma2=np.log(0.4)))
model.compute(t, yerr)

def lnprob(p):
    model.set_parameter_vector(p)
    return model.log_likelihood(y, quiet=True) + model.log_prior() # log_prior() is a constant, more complicated
# priors go into model


import emcee

initial = model.get_parameter_vector()
ndim, nwalkers = len(initial), 32
p0 = initial + 1e-8 * np.random.randn(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

print("Running burn-in...")
p0, _, _ = sampler.run_mcmc(p0, 500)  # Run the mcmc to get the burn in time
sampler.reset()                      # forget everything you know

print("Running production...") # rerun without burn in since you now know where to start
sampler.run_mcmc(p0, 1000);
