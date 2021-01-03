#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 22:47:41 2019

@author: zhouge
"""
config = {
          "coordinatewise": {
              "net": "CoordinateWiseDeepLSTM",
              "net_options": {
                  "layers": (20, 20),
                  "preprocess_name": "LogAndSign",
                  "preprocess_options": {"k": 5},
                  "scale": 0.01,
              }}}
              
key = next(iter(config))
kwargs = config[key]
keys = [key]


a=[1,2,3]
b=[2,3,4]
print(*[b-a])
c=[*zip(a,b)]
print(c[0])
