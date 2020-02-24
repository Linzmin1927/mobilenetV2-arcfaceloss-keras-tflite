#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 将训练好的h5文件转为pb文件
import tensorflow as tf
from tensorflow.python.tools import freeze_graph

import os
import os.path as osp

import datetime
from face_loss_k import ArcFace



if __name__ == '__main__':
    #img_path = './chess0'
    #img_list = os.listdir(img_path)

    #路径参数
    input_path = './'
    weight_file = './checkpoint/20200223-172100-model_01-0.00.h5'
    json_file = './model.json'
    weight_file_path = osp.join(input_path,weight_file)
    output_graph_name = weight_file[:-3] + '.pb'


    # 输出
    output_dir = osp.join(os.getcwd(),"trans_model")

    # 将学习率置为0，避免BN层错误
    tf.keras.backend.clear_session()
    tf.keras.backend.set_learning_phase(0)

    # 加载模型
    file = open(json_file, 'r')
    model_json1 = file.read()
    model = tf.keras.models.model_from_json(model_json1,custom_objects={'ArcFace': ArcFace})
    model.load_weights(weight_file_path)    
    model.summary()

   
    print("===>input:")
    for out in model.inputs:
        print(out.op.name)
    print("===>output:")
    for out in model.outputs:
        print(out.op.name)


    save_dir = "./tmp_{:%Y-%m-%d_%H%M%S}".format(datetime.datetime.now())
    tf.saved_model.simple_save(tf.keras.backend.get_session(),
                                 save_dir,
                                 inputs={"input_1": model.inputs[0]},
                                 outputs={'embeddings': model.outputs[1]}) # 确定embeddings输出是哪一个
    
    #  print("output:",model.outputs[0].op.name,model.outputs[1].op.name, model.outputs[2].op.name )
    freeze_graph.freeze_graph(None,
                              None,
                              None,
                              None,
                              model.outputs[1].op.name,
                              #  'embeddings,output',
                              None,
                              None,
                              os.path.join(save_dir, "frozen_model.pb"),
                              False,
                              "",
                              input_saved_model_dir=save_dir)
    print('coverted to PB .....final!!')


