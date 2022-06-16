# -*- coding: utf-8 -*-
"""
Created on Mon May 23 01:49:12 2022

@author: artur
"""

import onnx
from onnxsim import simplify

onnx_model = onnx.load("C:/Users/Artur/Desktop/Drone Project/WCrobarmPackPython/arturnet.onnx")
model_simpified, check = simplify(onnx_model)
onnx.save(model_simpified, "C:/Users/Artur/Desktop/Drone Project/WCrobarmPackPython/arturnetsimp.onnx")