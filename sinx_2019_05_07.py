# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 22:51:47 2019

@author: GeDongYuan
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
example_p = 360
xs = np.linspace(-3.14, 3.14, example_p)
ys = np.sin(xs) + np.random.uniform(-0.3, 0.3, example_p)
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')
W_1 = tf.Variable(tf.random_normal([1]), name='W_1')
W_2 = tf.Variable(tf.random_normal([1], name='W_2'))
W_3 = tf.Variable(tf.random_normal([1]), name='W_3')
b = tf.Variable(tf.random_normal([1]), name='b')
y_ = tf.add(tf.multiply(X,W_1),b)
y_ = tf.add(y_, tf.multiply(tf.pow(X,2),W_2))
y_ = tf.add(y_, tf.multiply(tf.pow(X,3),W_3))
n_samples = xs.shape[0]
with tf.name_scope("LOSS"):
    loss = tf.reduce_sum(tf.square(Y-y_))/n_samples
tf.summary.scalar('LOSS_GXSUT',loss)
sess=tf.Session()
merged = tf.summary.merge_all()
log_writer = tf.summary.FileWriter("./JEEE", sess.graph) 
learning_rate =  0.01299
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range (501):
        total_loss = 0
        for x,y in zip(xs, ys):
            _,l = sess.run([optimizer, loss],feed_dict={X:x, Y:y})
            total_loss += l
        if i%100 == 0:
            gdy= sess.run(merged,feed_dict={X:x, Y:y})
            print('Epochs={0}, average_loss={1}, learning_rate={2}'
                  .format(i,  l,  learning_rate))
            print("W_1=",sess.run(W_1),",  W_2=",sess.run(W_2),",  W_3=", 
                  sess.run(W_3),",  b=",sess.run(b))
            log_writer.add_summary(gdy,i)
    log_writer.close()
    W_1,W_2,W_3,b = sess.run([W_1,W_2,W_3,b]) 
plt.plot(xs,ys,'g.',label='Real Data')
plt.plot(xs,xs*W_1+np.power(xs,2)*W_2+np.power(xs,3)*W_3+b,'r-',lw=5,
         label='Predicted Data')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sine Function Fitting, Developed by GXUST') 
plt.show()