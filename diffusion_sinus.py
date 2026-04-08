#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 09:36:15 2026

@author: pauline
"""
import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(1,100,1000)
plt.plot(x, np.sin(x))
plt.plot(x, 0.7*np.sin(x))

plt.savefig('diffusion_sinus.pdf')

