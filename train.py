import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import warnings
warnings.filterwarnings("ignore")

import gc 
import cv2
import time
import numpy as np
import prettytable as pt

from sklearn.metrics import roc_curve, auc
from absl import app as absl_app  #
from absl import flags

from dataset import  batch_generator, load_dataset
from evaluate_data import load_evaluate_data,predict_evaluate_data
from verification import evaluate 
from face_loss_k import ArcFace

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2  import InceptionResNetV2
from tensorflow.keras.applications.xception import Xception
from my_Mobilenet import MobileNetV2

config = tf.ConfigProto()
#  config.gpu_options.per_process_gpu_memory_fraction = 0.95 #比例
config.gpu_options.allow_growth = True #按需

l = tf.keras.layers
FLAGS = flags.FLAGS  #

flags.DEFINE_integer('class_number', 85742, 'class number.')
flags.DEFINE_string('train_data', '../faces_emore/train.tfrecords', 'train data(tfrecode file)')

flags.DEFINE_integer('embedding_size', 512, 'embedding feature size .')
flags.DEFINE_float('learning_rate', 0.1, 'learning rate.')
flags.DEFINE_integer('img_size', 112, 'image size w==h.')
flags.DEFINE_float('arcface_s', 64, 'arcface loss s.')
flags.DEFINE_float('arcface_m', 0.5, 'arcface loss m.')
flags.DEFINE_list('lr_scheduler', [13,26], 'lr_scheduler')

flags.DEFINE_string('output_dir', "./output", 'save path.')
flags.DEFINE_string('checkpoint', './checkpoint', 'model save path')
flags.DEFINE_string('valid_dir', './valid_dataset/', 'valid data dir')
flags.DEFINE_string('logs_dir', './logs/', 'logs info dir')
#  flags.DEFINE_string('fine_tuning',  './checkpoint/20191221-154909-model_342-0.72.h5', 'fine tuning model')
flags.DEFINE_string('fine_tuning', '', 'fine tuning model')

flags.DEFINE_integer('batch_size', 8, 'batch size.')
flags.DEFINE_integer('epochs', 600, 'epochs.')
#  flags.DEFINE_integer('steps_per_epoch', 20000, 'epochs.')
flags.DEFINE_integer('steps_per_epoch', 100, 'epochs.')
flags.DEFINE_float('norm_loss', 2e-5, 'L1 _loss.')
flags.DEFINE_float('decay', 2e-5, 'weight_decay.')

def build_model():
    #  主干网络
    #  base_model = InceptionV3(
    base_model = MobileNetV2(
    #  base_model = Xception(
    #  base_model =InceptionResNetV2(
        alpha = 1.8,
        include_top=False,  # 是否包括顶层全连接
        #  weights='imagenet',  # 预训练权重
        weights=None,  # 预训练权重
        input_tensor=None,  # 输入张量，layers.Input()
        input_shape=(FLAGS.img_size, FLAGS.img_size, 3),  # 输入张量shape，仅在include_top为False时有效.宽高不小于75
        pooling=None,  # 仅在include_top为False时有效
        classes=None)  # 仅在include_top为True时，且weights未被指定时生效
    x = base_model.output
    #  out_shape = x.shape[1:3]
    out_shape = x.get_shape().as_list()[1:3]
    x = l.DepthwiseConv2D(kernel_size=out_shape,strides=(1,1))(x)
    x = l.PReLU()(x)
    x = l.BatchNormalization(axis = -1)(x)
    x = l.Conv2D(FLAGS.embedding_size, (1,1) )(x)
    x = l.PReLU()(x)
    #  x = l.BatchNormalization(axis= -1, epsilon=1e-3)(x)
    base_logits = l.Flatten(name = 'l2_loss')(x)
    embeddings = l.Lambda(lambda t: tf.nn.l2_normalize(t, 1, 1e-10), name='embeddings')(base_logits)
    embedder = tf.keras.models.Model(base_model.input, embeddings)
   
    # arcface_loss
    train_label = l.Input(shape=(), dtype="int32", name="train_label")
    predict_logits =  ArcFace( n_classes= FLAGS.class_number ,
                               s= FLAGS.arcface_s,
                               m= FLAGS.arcface_m,
                               name="arcface_loss_out",
                               #  regularizer= tf.keras.regularizers.l2(2e-5)
                               regularizer=None
                              )([embedder.output, train_label])

    sess = tf.keras.backend.get_session()
    sess.run(tf.global_variables_initializer())

    arcface_model = tf.keras.models.Model([embedder.input, train_label],
                                          [predict_logits, embeddings , base_logits],
                                          name="arcface_model")
    
    # 两个输入：embedder.input : 即主干网络的输入,就是归一化的图片
    #          train_label  : 图片训练的标签，训练完成后丢弃
    # 四个输出：predict_logits: arcfaceloss输出，训练完成后丢弃
    #          embeddings:   主干网提取的特征向量，我们想要的,实际应用中想要的
    #          base_logits:  主干网输出，用于计算l2 loss,训练完成后丢弃


    return arcface_model


def loss_regual(labels, prelogits):
    '''
    计算正则化化loss
    '''
    eps = 1e-5
    prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(prelogits) + eps, ord=1, axis=1))
    
    return prelogits_norm

def loss_arcface(labels, logit):
    '''
    '''
    arcface_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=tf.cast(tf.reshape(labels,(-1,)),tf.int32) ))
    
    return arcface_loss



import matplotlib.pyplot as plt
class ValidCallback(tf.keras.callbacks.Callback):
    '''自定义回调函数，用于每隔epoch结束后在验证集上验证模型效果 '''

    def __init__(self):
        
        datadirs = os.listdir(FLAGS.valid_dir)
        self.datadirs = datadirs
        self.sess = tf.keras.backend.get_session()
    
    def on_epoch_end(self, epoch, logs=None): 
        result = pt.PrettyTable( ["data set", "AUC", "ACC" ,"VR @ FAR ", "dist max", "dist min"])
        val_all = []
        auc_all = []
        acc_all = []
        dist_all = []
        for i in range(len(self.datadirs)) : 
            path = FLAGS.valid_dir + "/" + self.datadirs[i]
            embeddings, issamelab = predict_evaluate_data(path, [FLAGS.img_size,FLAGS.img_size], self.model, FLAGS.embedding_size)
            
            embeddings1 = embeddings[0::2]
            embeddings2 = embeddings[1::2]
            diff = np.subtract(embeddings1, embeddings2)
            dist = np.sum(np.square(diff), 1)
            del embeddings1,embeddings2
            fpr, tpr,ths = roc_curve(np.asarray(issamelab).astype('int'),dist, pos_label=0 )
            auc_score = auc(fpr, tpr)
            #  print('\ndataset:',self.datadirs[i])
            #  print('embed:',np.max(embeddings),np.min(embeddings))
            #  print("dist:",np.max(dist),np.min(dist))
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
            result.add_row([self.datadirs[i] ,
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
        
        logs['predict_labels_validacc'] = np.mean(np.array(val_all))
        logs['max_dis'] =  np.mean(np.array(dist_all),0)[0]
        logs['min_dis'] =  np.mean(np.array(dist_all),0)[1]
        logs['auc'] =  np.mean(np.array(auc_all))
        print(result)

def get_callbacks(strname,model):
    
    time_str = time.strftime("%Y%m%d-%H%M%S", time.localtime()) 
    # checkpoint,earlystot,tensorboard
    save_dir = FLAGS.checkpoint
    filepath =time_str + "-model_{epoch:02d}-{%s:.2f}.h5" % (strname)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(save_dir, filepath),
                                                    monitor='loss',
                                                    verbose=0,
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    mode='auto',
                                                    period=1)

    stop = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                            min_delta=0,
                                            patience=30,  # 连续多少个echo没有达到更高精度就退出
                                            verbose=0,
                                            mode='auto')

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=FLAGS.output_dir)
    def scheduler(epoch):
    # 每隔135个epoch，学习率减小为原来的1/10
        lr = FLAGS.learning_rate
        for i,epoch_th in enumerate(FLAGS.lr_scheduler):
            if epoch == epoch_th:
                lr = lr / (10**i)
                print("lr changed to {}".format(lr))
        return lr
    
    reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
    #  reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=10, mode='auto')
    filename = FLAGS.logs_dir+ time_str + ".log"
    logger = tf.keras.callbacks.CSVLogger(filename, separator=',', append=False)
    callbacks = [reduce_lr, checkpoint, stop,logger, tensorboard]
    #  callbacks = [ checkpoint, stop,logger, tensorboard]
    return callbacks

def model_train(model):
    callback = get_callbacks("predict_labels_validacc",model)
    # 自定义验证
    my_callback = ValidCallback()
    callbacks = [ my_callback] + callback
    print('!!!!!', model.metrics_names)

    tfrecode_file = FLAGS.train_data
    batch_size=FLAGS.batch_size
    n_classes=FLAGS.class_number
    suffle_buffer=5000
    train_gen = batch_generator(tfrecode_file,batch_size,suffle_buffer)
    steps_per_epoch=int((5822652 / batch_size)/30)
    model.fit_generator( train_gen,
                    #  steps_per_epoch=30,
                    #  steps_per_epoch=steps_per_epoch,
                    steps_per_epoch=FLAGS.steps_per_epoch,
                    epochs=FLAGS.epochs,
                    callbacks=callbacks,
                    verbose=1)


def train(model):
    
    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)
    if not os.path.exists(FLAGS.checkpoint):
        os.mkdir(FLAGS.checkpoint)
    if not os.path.exists(FLAGS.logs_dir):
        os.mkdir(FLAGS.logs_dir)


    losses = {
        'arcface_loss_out': 'sparse_categorical_crossentropy',
        #  'arcface_loss_out': loss_arcface,
        #  'embeddings': loss_regual,
        'l2_loss': loss_regual
    }
    loss_wights = [1, 0,  FLAGS.norm_loss]

    metric = {
        'arcface_loss_out': 'sparse_categorical_accuracy',
        #  'output': 'sparse_categorical_accuracy',
    }

    #  optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
    #  optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate,decay= FLAGS.decay , epsilon=0.1)
    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate,decay= 0, epsilon=0.1)
    model.compile(optimizer=optimizer,
                  loss=losses,
                  loss_weights=loss_wights,
                  metrics=metric
                  )

    if FLAGS.fine_tuning != "":
        model.load_weights(FLAGS.fine_tuning)
        print("!!pretrain model loading:%s" % (FLAGS.fine_tuning))
    #  loss_debug(model)
    model_train(model)



def main(unused_argv):

    facenet = build_model()
    # 保存模型结构
    model_json = facenet.to_json()
    with open(r'model.json', 'w') as file:
        file.write(model_json)
    #  facenet = tf.keras.utils.multi_gpu_model(facenet, gpus=2)
    
    facenet.summary()
    train(facenet)


if __name__ == '__main__':
    absl_app.run(main)



