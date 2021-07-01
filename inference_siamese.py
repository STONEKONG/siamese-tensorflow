import tensorflow as tf 
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import cv2
import numpy as np
from glob import glob 
import os

from siamese_net import siamese_loss, data_load

from tools import eval_classfier

def inference(image_list, pb_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # sess = tf.Session()

    with gfile.FastGFile(pb_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='') # 导入计算图
    sess.run(tf.global_variables_initializer())

    input_1 = sess.graph.get_tensor_by_name('input_x1:0')
    
    output = sess.graph.get_tensor_by_name('siamese/output:0')

    output_image = sess.run(output, feed_dict={input_1:image_list})

    return output_image

if __name__ == "__main__":
    import random
    
    pb_path = 'pb/siamese.ckpt-796-10.11599.pb'
    input_dir = '/home/xiaozhengheng/nfs-75/xiaozhiheng/code/VI/VI_detector/cuted_img/test_new'
    input_dir_ok = '/home/xiaozhengheng/nfs-75/xiaozhiheng/code/VI/VI_detector/cuted_img/standard_ok'
    batch_size = 9
    image_shape = 60
    thresh = 15
    data_1, label_1 = data_load(input_dir, image_shape)

    data_2, label_2 = data_load(input_dir_ok, image_shape)

    # NG为0，OK为1
    # y_list = np.array(label_1==label_2, dtype=np.float32)

    # len_batch_idx = int(len(label_1) // batch_size)
    num_data_1 = len(label_1)
    num_data_2 = len(label_2)

    predict = []
    label = []

    batch_len = int(num_data_1 / batch_size)
    for idx in range(batch_len):
        
        batch_data_1 = data_1[idx*batch_size:(idx+1)*batch_size]
        batch_label_1 = label_1[idx*batch_size:(idx+1)*batch_size]

        ind_2 = random.sample(range(0, num_data_2), batch_size)
        batch_data_2 = data_2[ind_2]
        batch_label_2 = label_2[ind_2]

        batch_y = np.array(batch_label_1==batch_label_2, dtype=np.float32)
        output_1 = inference(batch_data_1, pb_path)
        output_2 = inference(batch_data_2, pb_path)

        diff = output_1 - output_2
        diff = np.sum(np.square(diff), 1)
        diff = np.sum(diff, 1)
        diff = np.sqrt(np.mean(diff, 1))
        
        pre = diff.copy()
        pre[diff>thresh] = 0
        pre[diff<=thresh] = 1
        predict.append(list(pre))
        label.append(list(batch_y))


    predict = np.squeeze(np.reshape(np.array(predict, np.int32), [1,-1]), 0)
    label = np.squeeze(np.reshape(np.array(label, np.int32), [1,-1]), 0)
    precision, recall, acc = eval_classfier(predict, label)
    print('loubao is:', 1 - precision)
    print('wubao is:', 1 - recall)
    print('acc is:', acc)
     