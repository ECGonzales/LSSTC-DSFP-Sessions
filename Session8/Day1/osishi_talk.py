import numpy as np
import matplotlib.pyplot as plt

# procdeural
def nsquares(n):
    squares = []
    for i in range(n):
        squares.append(i**2)
        return squares

nsquares(10)

# Functional
sq = lambda n: [i**2 for i in range(n)]
print(sq(10))


# ---- Example of a class and objects ----
# a method if just s function defined inside a class
class Observation():  # "object" and "class" are interchangable
    def __init__(self, data):  # method-- a function inside a class. This one is the constructor
        self.data = data  # attribute

    def average(self):  # method
        dsum = 0
        for i, d in enumerate(self.data):
            dsum += d
        average = dsum / (i+1)
        return average

obs1 = Observation([0, 1, 2])
obs2 = Observation([4, 5, 6])

print("Avg 1 = {:e}; Avg2 = {:e}".format(obs1.average(), obs2.average()))  # :e - scientific format
print("Type of Avg 1 = {:}; Type of Avg 2 = {:}".format(type(obs1), type(obs2)))

print(obs1.data)
print(obs2.data)

# Inheritance

class TimeSeries(Observation):  # this inherits all the properties that the Observation class had
    def __init__(self, time, data):
        self.time = time
        Observation.__init__(self, data)  # this calls the constructor of the base class
        #  super could replace Observation and call the thing in the parenthesis of the class
        if len(self.time) != len(self.data):
            raise ValueError("Time and data must have the same length")  # ValueError is a class

    def stop_time(self):  # this is a new constructor
        return self.time[-1]

# The TimeSeries class has average, time, and data
# self binds to the instance object

tobs = TimeSeries([0, 1, 2], [3, 4, 5])
print(tobs)
print("Stop time = {:e}".format(tobs.stop_time()))  # stop_time needs () but nothing inside of it since it takes just self
print("tobs average = {:e}".format(tobs.average()))


# Objects in practice. In python everything is an object
print(print)
# output: <built-in function print>
dont_do_this = print  # this is an object representing a function
dont_do_this("dont do this!")


# Object oriented library in Matplotlib
x = np.linspace(0, 2*np.pi, 1000)
y_theory = np.sinc(x)
y = y_theory + np.random.rand(1000)

fig = plt.figure(figsize=(8, 8))  # create a figure object
ax_data = fig.add_axes([0.1, 0.4, 0.8, 0.8])
ax_residual = fig.add_axes([0.1, 0.1, 0.8, 0.3])  #

# this is one axis
ax_data.plot(x, y, label='sinc(x)')
ax_data.legend()
ax_data.set_ylabel('f(x)')  # the labels are attributes, set_ylabel is needed in object oriented

# This is the residual plot
ax_residual.plot(x, y-y_theory, label='residual')
ax_residual.legend()
ax_residual.set_xlabel('x')
ax_residual.set_ylabel('residual')

# an attribute with __ i.e average__ means don't fuck with this attribute for reals. __ won't have it appear in the list
# when hitting tab. pep8 wants them before i.e .__average


# Maybe I want to make a class for the similar figures in papers.
