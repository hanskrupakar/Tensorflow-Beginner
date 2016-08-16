import tensorflow as tf

with tf.variable_scope("addmul"):
    i1 = tf.placeholder(tf.float32)
    i2 = tf.placeholder(tf.float32)
    i3 = tf.placeholder(tf.float32)

intermed = tf.add(i2,i3)

op = tf.mul(intermed,i1)

with tf.Session() as sess:
    result = sess.run([intermed, op], feed_dict={i1:[5.0],i2:[15.0],i3:[25.0]})
    print result
