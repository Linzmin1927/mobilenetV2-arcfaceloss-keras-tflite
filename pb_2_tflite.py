#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import os
root_path = os.getcwd()
pb_dir = "./mobilenet_arcface_optimized.pb"
#输入模型节点名
input_arrays = ['input_1']
#输出节点名
output_arrays = ['embeddings/l2_normalize']
# 输入节点变量shape
input_shapes = {'input_1': [1,112,112,3]}
#从模型创建TFLiteConverter类
converter = tf.lite.TFLiteConverter.from_frozen_graph(pb_dir, input_arrays, output_arrays,input_shapes)
#将转换器的训练后是否量化设置为 true
converter.post_training_quantize=True
#允许自定义操作（之转换格式，就注释上面converter.post_training_quantize=True，打开converter.allow_custom_ops=True）
# converter.allow_custom_ops=True
# 8位量化
#  converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
#  input_arrays = converter.get_input_arrays()
#  converter.quantized_input_stats = {input_arrays[0] : (0., 1.)}  # mean, std_dev
#基于实例变量转换TensorFlow GraphDef（转换）
tflite_model=converter.convert()
#保存tflite地址
open("./mobilenet_arcface_optimized.tflite", "wb").write(tflite_model)












