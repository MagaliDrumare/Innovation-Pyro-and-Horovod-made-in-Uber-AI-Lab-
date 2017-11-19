#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 17:30:01 2017

@author: magalidrumare
"""
# import some dependencies
import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist


mu = Variable(torch.zeros(1))   # mean zero
sigma = Variable(torch.ones(1)) # unit variance
x = dist.normal(mu, sigma)      # x is a sample from N(0,1)
print(x)

log_p_x = dist.normal.log_pdf(x, mu, sigma)
print(log_p_x)

x = pyro.sample("my_sample", dist.normal, mu, sigma)
print(x)


def weather():
    cloudy = pyro.sample('cloudy', dist.bernoulli,
                         Variable(torch.Tensor([0.1])))
    cloudy = 'cloudy' if cloudy.data[0] == 1.0 else 'sunny'
    
    mean_temp = {'cloudy': [55.0], 'sunny': [85.0]}[cloudy]
    
    sigma_temp = {'cloudy': [10.0], 'sunny': [15.0]}[cloudy]
    
    temp = pyro.sample('temp', dist.normal,
                       Variable(torch.Tensor(mean_temp)),
                       Variable(torch.Tensor(sigma_temp)))
    return cloudy, temp.data[0]

for _ in range(10):
    print(weather())
    
def ice_cream_sales():
    cloudy, temp = weather()
    expected_sales = [200] if cloudy == 'sunny' and temp > 60.0 else [50]
    ice_cream = pyro.sample('ice_cream', dist.normal,
                            Variable(torch.Tensor(expected_sales)),
                            Variable(torch.Tensor([10.0])))
    return ice_cream
print(ice_cream_sales())
   
  