#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 强制cpu计算
import tensorflow as tf
import mxnet as mx
from sklearn.metrics import roc_curve, auc
import prettytable as pt
from tqdm import tqdm
import numpy as np
import cv2
import pickle
import gc 
from verification import evaluate 

TF_DIR = "./mobilenet_arcface_optimized.tflite" 
VALID_DIR = "./valid_dataset"
class FACENET_TFLITE:
    def __init__(self, model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        print(str(input_details))
        output_details = interpreter.get_output_details()
        print(str(output_details))
        self.inputs = input_details
        self.outputs = output_details
        self.interpreter = interpreter 

    def get_embedding(self, images):
        self.interpreter.set_tensor(self.inputs[0]['index'], images) 
        self.interpreter.invoke()
        embedding = self.interpreter.get_tensor(self.outputs[0]['index'])

        return embedding


def predict_evaluate_data_TFLITE(db_name, image_size , model, embedsize = 512):    
    batch = 100
    bins, issame_list = pickle.load(open(os.path.join(db_name), 'rb'), encoding='bytes')
    datasets = np.zeros( (batch , image_size[0], image_size[1], 3))
    embedings = np.zeros((len(issame_list)*2, embedsize))
    count = 0
    
    for i in tqdm(range(len(issame_list)*2),desc="evalutaing %s ..."%db_name) :
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        del _bin
        # img = cv2.imdecode(np.fromstring(_bin, np.uint8), -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img - 127.5
        img = img * 0.0078125
        datasets[count, ...] = img 
        count += 1
        i = i+1
        if i % batch  == 0:
            count = 0
            #  print('loading bin', i)
            bz_data = datasets[0:batch,...]
            bz_embedings = np.zeros([len(bz_data),512])
            for ii in range(len(bz_data)):
                em = model.get_embedding(np.expand_dims(bz_data[ii],axis = 0).astype('float32'))
                bz_embedings[ii] = em
            embedings[i-batch:i,...] = bz_embedings
    else:
        ll = i % batch 
        bz_embedings = np.zeros([ll,512])
        if ll !=0:
            ll_in = i // batch
            bz_data = datasets[0:ll,...]
            for ii in range(len(bz_data)): 
                em = model.get_embedding(np.expand_dims(bz_data[ii],axis = 0))
                bz_embedings[ii] = em
            embedings[ll_in*batch : ll_in*batch+ll: ,...] = bz_embedings
    del bins 
    gc.collect()
    return embedings, issame_list

def tflite_model_test(model_dir): 
    model = FACENET_TFLITE(model_dir)
    img_size = 112 
    valid_dir = VALID_DIR
    datadirs = os.listdir(valid_dir)
    embedding_size = 512

    result = pt.PrettyTable( ["data set", "AUC", "ACC" ,"VR @ FAR ", "dist max", "dist min"])
    val_all = []
    auc_all = []
    acc_all = []
    dist_all = []
    for i in range(len(datadirs)) : 
        path = valid_dir + "/" + datadirs[i]
        embeddings, issamelab = predict_evaluate_data_TFLITE(path, [img_size,img_size], model, embedding_size)
            
        embeddings1 = embeddings[0::2]
        embeddings2 = embeddings[1::2]
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)
        del embeddings1,embeddings2
        fpr, tpr,ths = roc_curve(np.asarray(issamelab).astype('int'),dist, pos_label=0 )
        auc_score = auc(fpr, tpr)
        if 0:
            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange',
                              lw=lw, label='ROC curve (area = %0.2f)' % auc_score)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right")
            #  plt.show()
            plt.savefig("roc.png")

        #  print('-----10 folds------')
        tpr, fpr, accuracy, val, val_std, far = evaluate(embeddings, issamelab)
        del embeddings,issamelab
        gc.collect()
        result.add_row([datadirs[i] ,
                         round(auc_score,2),
                         "%1.3f+-%1.3f" % (np.mean(accuracy), np.std(accuracy)),
                         "%2.5f+-%2.5f @ FAR=%2.5f" % (val, val_std, far),
                         round(np.max(dist),2),
                         round(np.min(dist),2),
                             ])
        val_all.append(val)
        acc_all.append(accuracy)
        auc_all.append(auc_score)
        dist_all.append([np.max(dist),np.min(dist)])
    
    print(result)



tflite_model_test(TF_DIR)




