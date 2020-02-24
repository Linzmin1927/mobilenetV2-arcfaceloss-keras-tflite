import tensorflow as tf
import numpy as np
import cv2
def decode_jpeg(image_buffer, scope=None):
  """Decode a JPEG string into one 3-D float image Tensor.
  Args:
    image_buffer: scalar string Tensor.
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor with values ranging from [0, 1).
  """
  with tf.name_scope(values=[image_buffer], name=scope,
                     default_name='decode_jpeg'):
    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height and width
    # that is set dynamically by decode_jpeg. In other words, the height
    # and width of image is unknown at compile-time.
    image = tf.image.decode_jpeg(image_buffer, channels=3)

    # After this point, all image pixels reside in [0,1)
    # until the very end, when they're rescaled to (-1, 1).  The various
    # adjust_* ops all require this range for dtype float.
    #  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image


def parse_function(example_proto):
    #在recoder里字符串数据转码为图片数据
    
    features = {'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)}
    features = tf.parse_single_example(example_proto, features)
    # You can do more image distortion here for training data

    img = tf.image.decode_jpeg(features['image_raw'])
    img = tf.reshape(img, shape=(112, 112, 3))
    #  print(img)
    channels = tf.unstack (img, axis=-1)
    img = tf.stack ([channels[2], channels[1], channels[0]], axis=-1)
 
    #img = tf.py_func(random_rotate_image, [img], tf.uint8)
    img = tf.cast(img, dtype=tf.float32)
    img = tf.subtract(img, 127.5)
    img = tf.multiply(img,  0.0078125)
    img = tf.image.random_flip_left_right(img)
    #  img = tf.cast(img,dtype=tf.float16)
    label = tf.cast(features['label'], tf.int64)
    
    return img,label


def load_dataset(input_path,batch_size,suffle_buffer):
    dataset = tf.data.TFRecordDataset(input_path)
    if  suffle_buffer is None:
        dataset = dataset.repeat()
    else:
        dataset = dataset.shuffle(suffle_buffer).repeat()
    dataset = dataset.map(parse_function)
    dataset = dataset.batch(batch_size).prefetch(suffle_buffer)
    #  dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator


def batch_generator(tfrecode_file,batch_size,suffle_buffer):
    iterator = load_dataset(tfrecode_file,batch_size,suffle_buffer)
    data = iterator.get_next()
    with tf.Session() as sess:
        #  sess.run(iterator.initializer)
        ret = sess.run(data)
        while True:
            ret = sess.run(data)
            X_batch, y1_batch = ret
            ret = ({'input_1': X_batch, 'train_label': y1_batch}, {'embeddings': None, 'arcface_loss_out': y1_batch, 'l2_loss':None})
            yield ret





if __name__ == '__main__' :
    #  tfrecode_file = './train-00000-of-00001'
    tfrecode_file = '../faces_emore/train.tfrecords'
    batch_size = 10
    suffle_buffer = 1000
    iterator = load_dataset(tfrecode_file,batch_size,suffle_buffer)
    
    xx = iterator.get_next()
    session = tf.Session()
    #  session.run(iterator.initializer)
    ret = session.run(xx)
    imgs,labs = ret
    for i in range(len(imgs)):
        print('lab:',labs[i])
        im = imgs[i]
        img = im*128+127.5
        img = img.astype('int')
        cv2.imwrite('img_%d.jpg'%i,img)
   
    print(imgs.shape)

    batch_size=64
    n_classes=85742
    suffle_buffer=10000

#
