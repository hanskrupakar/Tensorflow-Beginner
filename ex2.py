import tensorflow as tf
import numpy as np

X = np.random.randint(1,10,[50000,10])
Y = np.reshape(np.sum(X,axis=1), (50000,1))

x = tf.placeholder(tf.float32, shape=(50000,10))
y = tf.placeholder(tf.float32, shape=(50000,1))

with tf.variable_scope("LinReg"):
    W_ih = tf.get_variable("weight_ih", (10,7), initializer=tf.random_normal_initializer())
    W_ho = tf.get_variable("weight_ho", (7,1), initializer=tf.random_normal_initializer())
    B_ih = tf.get_variable("bias_ih", (50000,7), initializer=tf.constant_initializer(0.0))
    B_ho = tf.get_variable("bias_ho", (50000,1), initializer=tf.constant_initializer(0.0))
    
    ip = tf.matmul(x,W_ih) + B_ih
    hidden = tf.tanh(ip)
    y_pred = tf.matmul(hidden,W_ho) + B_ho
    loss = tf.reduce_sum((y-y_pred)**2/50000)
    
    optimizer = tf.train.AdamOptimizer()
    
    grad_desc = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    _,loss_val,p = sess.run([grad_desc, loss, y_pred], feed_dict={x:X, y:Y})
    for i in xrange(5000):
        if(i%50==0):  
            print("LOSS: %0.2f" % loss_val)
        _,loss_val,p = sess.run([grad_desc, loss, y_pred], feed_dict={x:X, y:Y})
        
    _,loss_val,p = sess.run([grad_desc, loss, y_pred], feed_dict={x:np.ones((50000,10)), y:10*np.ones((50000,1))})
    print("TEST INPUT: ")
    print(np.zeros((50000,10)))
    print("TEST OUTPUT: ")
    print(p)
    print("LOSS IN TRAINING: %s" % loss_val)
    