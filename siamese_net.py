
import tensorflow as tf
import argparse
import numpy as np
import time
import os
from layers import *
from tools import data_load_2

def siamese_loss(out1, out2, y, M=50, alpha=1.0):
    E_w = tf.sqrt(tf.reduce_sum(tf.square(out1 - out2), [1,2,3]))
    pos = tf.multiply(y, E_w)
    neg = tf.multiply(1-y, tf.maximum(M-E_w, 0))
    pos_loss = tf.reduce_mean(pos)
    neg_loss = tf.reduce_mean(neg)
    total_loss = pos_loss + neg_loss 
    return total_loss, pos_loss, neg_loss


def Siamese(inputs, name, reuse):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        conv1_1 = conv2d(inputs, 5, 16, 1, normal=True, activation='relu', name='conv1')
        conv1_2 = conv2d(conv1_1, 3, 32, 1, normal=True, activation='relu', name='conv2')
        pool1 =  tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')

        conv2_1 = conv2d(pool1, 3, 64, 1, normal=True, activation='relu', name='conv3')
        conv2_2 = conv2d(conv2_1, 3, 64, 1, normal=True, activation='relu', name='conv4')
        pool2 =  tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')

        conv3_1 = conv2d(pool2, 3, 32, 1, normal=True, activation='relu', name='conv5')
        conv3_2 = conv2d(conv3_1, 3, 32, 1, normal=True, activation='relu', name='conv6')
        pool3 =  tf.nn.max_pool(conv3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool3')

        conv4_1 = conv2d(pool3, 3, 16, 1, normal=True, activation='relu', name='conv7')
        conv4_2 = conv2d(conv4_1, 1, 1, 1, normal=True, activation='None', name='conv8')
        
        pool4 =  tf.nn.max_pool(conv4_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool4')
        return pool4


def train(args):
    model_time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
    save_path = 'checkpoints/' + model_time
    log_dir = 'log/' + model_time

    if not os.path.exists(save_path):
        os.makedirs(save_path)
   
    ###train_data
    train_data_1, label_1 = data_load_2(args.train_data_dir, args.img_size)
    train_data_2, label_2 = data_load_2(args.train_data_dir, args.img_size)
    y_list = np.array(label_1==label_2, dtype=np.float32)
    ###test_data
    test_data_1, test_label_1 = data_load_2(args.test_data_dir, args.img_size)
    test_data_2, test_label_2 = data_load_2(args.test_data_dir, args.img_size)
    test_y_list = np.array(test_label_1==test_label_2, dtype=np.float32)
    iterations = int(len(y_list) / args.batch_size)
    iterations_test = int(len(test_y_list) / args.batch_size)
    x_input_1 = tf.placeholder(tf.float32, shape=[None, args.img_size, args.img_size, 3], name='input_x1')
    x_input_2 = tf.placeholder(tf.float32, shape=[None, args.img_size, args.img_size, 3], name='input_x2')
    y = tf.placeholder(tf.float32, shape=[None], name='y')

    
    out1 = Siamese(x_input_1, 'siamese', False)
    out2 = Siamese(x_input_2, 'siamese', True)

    with tf.name_scope('leaning_rata'):
        global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
        warmup_steps = tf.constant(args.warmup_epoch * iterations, dtype=tf.float64, name='warmup_steps')
        train_steps = tf.constant(args.max_epoch * iterations, dtype=tf.float64, name='train_steps')
        lr = tf.cond(
            pred=global_step < warmup_steps,
            true_fn=lambda: global_step / warmup_steps * args.init_lr,
            false_fn=lambda: args.init_lr + 0.5 * (args.init_lr - args.end_lr) *
                                (1 + tf.cos((global_step - warmup_steps) / (train_steps - warmup_steps) * 0.5 * np.pi))
        )
        global_step = tf.assign_add(global_step, 1.0)
    with tf.variable_scope('metrics'):
        loss = siamese_loss(out1, out2, y, M=args.up_boundary)
        # optimizer = tf.train.AdamOptimizer(lr).minimize(loss[0], global_step=global_step)
        if args.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(lr).minimize(loss[0], global_step=global_step)
        elif args.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss[0], global_step=global_step)

    with tf.name_scope('summary'):
        loss_ave = tf.Variable(0, dtype=tf.float32, trainable=False)
        loss_summary = tf.summary.scalar('loss', loss_ave)
        lr_summary = tf.summary.scalar('lr', lr)
        merged_summary = tf.summary.merge_all()

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=args.max_epoch)
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    with tf.Session(config=config) as sess:
        writer_train = tf.summary.FileWriter(log_dir + '/train', sess.graph)
        writer_test = tf.summary.FileWriter(log_dir + '/val')
        sess.run(tf.global_variables_initializer())
        for epoch in range(args.max_epoch):
            print_train_loss = np.zeros(3, dtype=np.float32)
            print_test_loss = np.zeros(3, dtype=np.float32)
            # for i in range(iterations):
            #     batch_data_1 =  train_data_1[i*batch_size:(i+1)*batch_size]
            #     batch_data_2 = train_data_2[i*batch_size:(i+1)*batch_size]
            #     batch_y = y_list[i*batch_size:(i+1)*batch_size]
            #     _, train_loss, g_s = sess.run([optimizer, loss, global_step],feed_dict={x_input_1:batch_data_1, 
            #                                 x_input_2:batch_data_2, y:batch_y})
            #     test_loss = sess.run(loss, feed_dict={x_input_1:test_data_1[0:2*batch_size], 
            #                                 x_input_2:test_data_2[0:2*batch_size], 
            #                                 y:tets_y_list[0:2*batch_size]})
            #     print_train_loss += train_loss
            #     print_test_loss += test_loss
            for i in range(iterations):

                batch_data_1 = train_data_1[i*args.batch_size:(i+1)*args.batch_size]
                batch_data_2 = train_data_2[i*args.batch_size:(i+1)*args.batch_size]
                batch_y = y_list[i*args.batch_size:(i+1)*args.batch_size]
                _, train_loss, g_s = sess.run([optimizer, loss, global_step], feed_dict={x_input_1:batch_data_1, 
                                            x_input_2:batch_data_2, y:batch_y})
                print_train_loss += train_loss
            
            for j in range(iterations_test):
                test_batch_data_1 = test_data_1[j*args.batch_size:(j+1)*args.batch_size]
                test_batch_data_2 = test_data_2[j*args.batch_size:(j+1)*args.batch_size]
                test_batch_y = test_y_list[j*args.batch_size:(j+1)*args.batch_size]
                test_loss = sess.run(loss, feed_dict={x_input_1:test_batch_data_1, 
                                                x_input_2:test_batch_data_2, 
                                                y:test_batch_y})
                print_test_loss += test_loss
            
            print_train_loss = print_train_loss / iterations
            print_test_loss = print_test_loss / iterations_test

            sess.run(tf.assign(loss_ave, print_train_loss[0]))
            train_summ = sess.run(merged_summary)
            writer_train.add_summary(train_summ, g_s)

            sess.run(tf.assign(loss_ave, print_test_loss[0]))
            test_summ = sess.run(loss_summary)
            writer_test.add_summary(test_summ, g_s)
            
            # train_data_1, label_1 = data_load_2(args.train_data_dir, args.img_size)
            # train_data_2, label_2 = data_load_2(args.train_data_dir, args.img_size)
            # y_list = np.array(label_1==label_2, dtype=np.float32)

            print('epoch is:{} train_loss is:{} train_pos_loss is:{} train_neg_loss is:{}'.format(epoch, print_train_loss[0], print_train_loss[1], print_train_loss[2]))
            print('             test_loss is:{}  test_pos_loss is:{}  test_neg_loss is:{}'.format(print_test_loss[0], print_test_loss[1], print_test_loss[2]))
            saved_model_name = os.path.join(save_path, 'siamese.ckpt-%d-%.5f' % (epoch, float(print_test_loss[0])))
            saver.save(sess, saved_model_name)

        writer_train.close()
        writer_test.close()
    

def parse():
    """
    parse params
    Returns:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_lr', help='initialized learning rate', type=float, default=5e-4)
    parser.add_argument('--end_lr', help='ended learning', type=float, default=5e-6)
    parser.add_argument('--warmup_epoch', help='', type=int, default=3)
    parser.add_argument('--optimizer', help='used optimizer, adam and sgd can be chosed', type=str, default='adam')
    parser.add_argument('--batch_size', help='batch size for training', type=int, default=4)
    parser.add_argument('--up_boundary', help='', type=int, default=60)
    parser.add_argument('--img_size', help='train,test sizes', type=int, default=128)
    parser.add_argument('--max_epoch', type=int, default=60)
    parser.add_argument('--train_data_dir', help='dir of train images', type=str, default='')
    parser.add_argument('--test_data_dir', help='dir of test images', type=str, default='')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse()
    train(args)