# Uber AI Lab 
* Uber AI Lab : http://uber.ai 
* Uber Engineering Blog : https://eng.uber.com

# What is  . ? 
* Horovod is a distributed training framework for TensorFlow. 
* The goal of Horovod is to make distributed Deep Learning fast and easy to use.
* Meet Horovod: Uber’s Open Source Distributed Deep Learning Framework for TensorFlow : https://eng.uber.com/horovod/
* Keras + Horovod = Distributed Deep Learning on Steroids : https://goo.gl/H3ixuD
* Horovod GitHub : https://github.com/uber/horovod

# What is Pyro ? 
* Pyro probabilistic programming language : http://pyro.ai
* Pyro is a tool for deep probabilistic modeling, unifying the best of modern deep learning and Bayesian modeling. 
* An intro to Probabilistic Programming with Ubers Pyro by Siraj Raval : https://youtu.be/ATaMq62fXno

# Simple Model 
* Read : http://pyro.ai/examples/intro_part_i.html#A-Simple-Model
* Install pytorch : http://pytorch.org
* Install pyro : http://pyro.ai  pip install pyro-ppl 

```python

# we have a bunch of data with daily mean temperatures and cloud cover.
# We want to reason about how temperature interacts with whether it was sunny or cloudy. 
# This model defines a probabilistic model that we can reason about using the techniques of probability theory. 
# For example we might ask: if I observe a temperature of 70 degrees, how likely is it to be cloudy? 

import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist


mu = Variable(torch.zeros(1))   # mean zero
sigma = Variable(todependenciesrch.ones(1)) # unit variance
x = dist.normal(mu, sigma)      # x is a sample from N(0,1)
print(x)

log_p_x = dist.normal.log_pdf(x, mu, sigma)
print(log_p_x)

x = pyro.sample("my_sample", dist.normal, mu, sigma)
print(x)


def weather():
    #according to this model 30% of the time it’s cloudy and 70% of the time it’s sunny.
    cloudy = pyro.sample('cloudy', dist.bernoulli,
                         Variable(torch.Tensor([0.3])))
    cloudy = 'cloudy' if cloudy.data[0] == 1.0 else 'sunny'
    
    #mean temperature is 55 degrees (Fahrenheit) on cloudy days and 75 degrees on sunny days.
    mean_temp = {'cloudy': [55.0], 'sunny': [75.0]}[cloudy]
    sigma_temp = {'cloudy': [10.0], 'sunny': [15.0]}[cloudy]
    temp = pyro.sample('temp', dist.normal,
                       Variable(torch.Tensor(mean_temp)),
                       Variable(torch.Tensor(sigma_temp)))
    return cloudy, temp.data[0]

for _ in range(3):
    print(weather())
    
    
def geometric(p, t=None):
    if t is None:
        t = 0
    x = pyro.sample("x_{}".format(t), dist.bernoulli, p)
    if torch.equal(x.data, torch.zeros(1)):
        return x
    else:
        return x + geometric(p, t+1)

print(geometric(Variable(torch.Tensor([0.5]))))

```

# First run 

```python
Variable containing:
-0.5981
[torch.FloatTensor of size 1]

Variable containing:
-1.0978
[torch.FloatTensor of size 1]

Variable containing:
-0.3139
[torch.FloatTensor of size 1]

('sunny', 82.88794708251953)
('sunny', 76.23553466796875)
('sunny', 61.52381134033203)

```




