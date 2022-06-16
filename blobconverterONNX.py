# -*- coding: utf-8 -*-
"""
Created on Mon May 23 02:45:41 2022

@author: artur
"""

import blobconverter

# blobconverter.from_onnx(
#     model="C:/Users/Artur/Desktop/Drone Project/WCrobarmPackPython/arturnetsimp.onnx",
#     output_dir="C:/Users/Artur/Desktop/Drone Project/WCrobarmPackPython/arturnetsimp.blob",
#     data_type="FP16",
#     #shaves=6,
#     use_cache=False,
#     optimizer_params=[]
# )

blob_path = blobconverter.from_openvino(
    xml="arturnetsimp.xml",
    bin="arturnetsimp.bin",
    data_type="FP32",
    shaves=4,
)
