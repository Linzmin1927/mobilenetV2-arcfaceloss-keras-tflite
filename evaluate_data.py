import mxnet as mx
import numpy as np
import pickle
import os
import cv2
import gc 
from tqdm import tqdm
# 此处加载处理好的验证集数据，该部分数据是采用mxnet格式存储的
def load_evaluate_data(db_name, image_size):
    bins, issame_list = pickle.load(open(os.path.join(db_name), 'rb'), encoding='bytes')
    datasets = np.empty((len(issame_list)*2, image_size[0], image_size[1], 3))

    for i in range(len(issame_list)*2):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        # img = cv2.imdecode(np.fromstring(_bin, np.uint8), -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img - 127.5
        img = img * 0.0078125
        datasets[i, ...] = img
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
            break
    print(datasets.shape)
    print("issame_list:",len(issame_list))

    return datasets, issame_list

def predict_evaluate_data(db_name, image_size , model, embedsize = 512):    
    batch = 50
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
            input_y_fake = np.zeros([batch , ])
            bz_data = datasets[0:batch,...]
            [_, bz_embedings,_] = model.predict([bz_data, input_y_fake ],verbose=0)
            embedings[i-batch:i,...] = bz_embedings
    else:
        ll = i % batch 
        if ll !=0:
            ll_in = i // batch
            input_y_fake = np.zeros([ll , ])
            bz_data = datasets[0:ll,...]
            [_, bz_embedings,_] = model.predict([bz_data, input_y_fake ],verbose=0)
            embedings[ll_in*batch : ll_in*batch+ll: ,...] = bz_embedings
    del bins 
    gc.collect()
    return embedings, issame_list






if __name__ == "__main__":
    binfile = './valid_dataset/lfw.bin'
    datasets, issame_list = load_evaluate_data(binfile,[112,112])
    print(datasets.shape)
