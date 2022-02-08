# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Programming in Python
# ## Exam: February 8, 2022
#
#
# You can solve the exercises below by using standard Python 3.9 libraries, NumPy, Matplotlib, Pandas, PyMC3.
# You can browse the documentation: [Python](https://docs.python.org/3.9/), [NumPy](https://numpy.org/doc/stable/user/index.html), [Matplotlib](https://matplotlib.org/3.5.1/contents.html), [Pandas](https://pandas.pydata.org/pandas-docs/version/1.2.5/), [PyMC3](https://docs.pymc.io/).
# You can also look at the [slides of the course](https://homes.di.unimi.it/monga/lucidi2021/pyqb00.pdf) or your code on [GitHub](https://github.com).
#
# **It is forbidden to communicate with others.** 
#

# %matplotlib inline
import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pymc3 as pm   # type: ignore

# ### Exercise 1 (max 3 points)
#
# The file [foxes.csv](foxes.csv) 
# lists data for 116 foxes from 30 different urban groups in England (data source is https://github.com/rmcelreath/rethinking). Group size (`groupsize`) varies from 2 to 8 individuals. Each group maintains its own (almost exclusive) urban territory. Some territories are larger than others. The `area` variable encodes this information.
# Some territories also have more `avgfood` than others. And food influences
# the `weight` of each fox. Read the data in a `DataFrame`, be sure `groupsize` has an `int8` data type and `group` a `category` data type. 

foxes = pd.read_csv('foxes.csv', sep=';', header=1, dtype={'group': 'category', 'groupsize': 'int8'})

foxes.dtypes

foxes.describe()

# ### Exercise 2 (max 2 points)
#
#
# Plot a histogram of `groupsize`.

fig, ax = plt.subplots(1, 1)
_ = ax.hist(foxes['groupsize'], bins=[b for b in range(foxes['groupsize'].min(), foxes['groupsize'].max()+1)])

# ### Exercise 3 (max 3 points)
#
#
# Plot a scatter plot of `area` vs. `groupsize`. Put a proper label on each axis.

fig, ax = plt.subplots(1, 1)
ax.scatter(foxes['area'], foxes['groupsize'])
ax.set_xlabel('area')
_ = ax.set_ylabel('groupsize')


# ### Exercise 4 (max 6 points)
#
# Write a function `standardize` to compute the standardized value of a (random) variable. The standardize value is the difference between the measured value and the mean value of the variable, divided by the standard deviation of the variable. For example, the standardized value for a variable with values `[1,-1,0]` is `[1.22474487, -1.22474487,  0.]`. To get the full marks, you should declare correctly the type hints (the signature of the function; you can use `np.typing.NDArray` for the type of a NumPy array) and add a doctest string.

def standardize(x: np.typing.NDArray) -> np.typing.NDArray:
    """Standard version of x.
    
    >>> all(abs(standardize(np.array([1,-1,0])) - np.array([1.22474487, -1.22474487,  0.])) < 1e-6)
    True
    """
    return (x - x.mean())/x.std()


# ### Exercise 5 (max 6 points)
#
# The Pearson's correlation coefficient between to (random) variables is the mean of the product of the deviations from the mean of each, divided by the product of the standard deviations. If $E[X]$ is the expected value (the mean) of a variable $X$ and $\sigma_X$ its standard deviation, the Pearson's $R_{xy}$ of two variables $x$ and $y$ is $R_{xy} = \frac{E[(x - E[x])\cdot(y - E[y])]}{\sigma_x\sigma_y}$.
#     
# Write a function to compute $R$. To get the full marks, you should declare correctly the type hints (the signature of the function; you can use `np.typing.NDArray` for the type of a NumPy array) and add a doctest string. Given `x` and `y`, the result of your function should be close (let's say with a difference less than `1e-6`) to the one returned by `np.corrcoef(x, y)[0,1]`.

def pearson_r(x: np.typing.NDArray, y: np.typing.NDArray) -> float:
    """Return Pearson's R.
    
    >>> x, y = np.array([1,2,3]), np.array([3,2,1])
    >>> abs(pearson_r(x, y) - np.corrcoef(x, y)[0,1]) < 1e-5
    True
    """
    return ((x - x.mean())*(y-y.mean())).mean()/(x.std()*y.std())


pearson_r(np.array([1,2,3]), np.array([3,2,1]))

# ### Exercise 6 (max 3 points)
#
# Add two columns to the foxes dataframe with the standardized values of `area` and `groupsize`.

foxes['std_area'] = standardize(foxes['area'])
foxes['std_groupsize'] = standardize(foxes['groupsize'])


# ### Exercise 7 (max 5 points)
#
# Plot a scatter plot of standardized `area` vs. standardized `groupsize`. Draw also the straight line passing from (0, 0) with a slope equal to the Pearson's coefficient between the two variables.

fig, ax = plt.subplots(1,1)
ax.scatter(foxes['std_area'], foxes['std_groupsize'])
x_area = np.linspace(foxes['std_area'].min(), foxes['std_area'].max(), 100)
ax.plot(x_area, pearson_r(foxes['std_area'], foxes['std_groupsize'])*x_area)
ax.set_xlabel('standardized area')
_ = ax.set_ylabel('standardized groupsize')


# ### Exercise 8 (max 5 points)
#
# Consider this statistical model: the standardized group size is proportional to the standardized area. Let's posit that group size follows a [normal](https://docs.pymc.io/en/v3/api/distributions/discrete.html#pymc3.distributions.continue.Normal) distribution with $\mu=\alpha\cdot A$ and standard deviation 1, where $A$ is the standardized area and $\alpha$ is the proportionality coefficient you want to estimate. Your *a priori* estimation of $\alpha$ is also normal with zero mean (and standard deviation 1). Use PyMC to sample the posterior distributions after having observed the actual values of the standardized group size.  Plot the results.

with pm.Model():
    alpha = pm.Normal('alpha', mu=0)
    groupsize = pm.Normal('groupsize', mu=alpha*foxes['std_area'], observed=foxes['std_groupsize'])
    trace = pm.sample(return_inferencedata=True)

_ = pm.plot_posterior(trace)
