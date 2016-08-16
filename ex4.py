import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn, rnn_cell

if __name__ == '__main__':
    X = np.random.randint(0,2,(50000,25,10))
    Y = np.reshape(np.sum(X,axis=2),(50000,25,1))
    
    X_test = np.random.randint(0,2,(1000,25,10))
    Y_test = np.reshape(np.sum(X_test,axis=2),(1000,25,1))
    
    W = tf.Variable(tf.random_normal([10,1], stddev=0.01)) 
    B = tf.Variable(tf.zeros([25,1]))
    
    x = tf.placeholder(tf.float32, [None,25,10])
    y = tf.placeholder(tf.float32, [None,25,1])
    
    lstm = rnn_cell.BasicLSTMCell(10,forget_bias=1.0)
    
    XT = tf.transpose(x, [1, 0, 2])
    XR = tf.reshape(XT, [-1, 10])
    X_split = tf.split(0, 25, XR)
    
    init_state = tf.placeholder("float", [None, 2*10])
    
    outputs, _states = rnn.rnn(lstm,X_split, init_state)
    
    res = tf.matmul(outputs[-1], W) + B
    
    print("XT: ", XT)
    print("XR: ",XR)
    print("X_SPLIT: ",X_split)
    print("RES: ", res.get_shape())
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(res, y))
    train_op = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in range(100):
            for start in range(0,50000,500):
                sess.run(train_op, feed_dict = {x: X[start:start+500], y: Y[start:start+500], init_state: np.zeros([50000, 20])})
        print(sess.run(outputs))        
        