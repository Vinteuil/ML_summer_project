##############################################
# Deep Convolutional Neural Network to       #
# discriminate between different comsologies #
# in the form of .fits images produced       #
# using the Full-sky Lognormal Astro-fields  #
# Simulation Kit (FLASK). To be run on       #
# a GPU instance in the cloud (specifically  #
# floydhub).                                 #
##############################################

!pip install astropy 
!pip install tensorflow
from astropy.io import fits
import numpy as np
from PIL import Image
import os
import tensorflow as tf 
from scipy import ndimage
from random import randint
import csv 
from sklearn.utils import shuffle
from numpy import inf

#initialising list for ellipticities
mod_e_list = [] 
#recording indices of fits files that present errors
bad_files = []  
count = 0

for filename in os.listdir('/my_data/'):
    print(filename)
    count = count + 1
    print(count)                        
    try: 
        #opening and storing arrays
        hdulist = fits.open('/my_data/' + str(filename))
        scidata = hdulist[1].data
        mod_e=np.sqrt((scidata["Q-pol"]**2)+(scidata["U-pol"]**2))
        mod_e_list.append(mod_e)
        
    
    except: 
        #recording bad files
        bad_files.append(count)
        pass


#creating a list of file names so in order to produce labels
#each file is labelled with a number. As for this simulation 
#data was produced in batches of 98 images (in two bins) 
#you can back track and create labels from their filenames
filename_list = [] 
count = 0 
for filename in os.listdir('/my_data/'):
    count = count + 1 
    if count not in bad_files: 
        filename_list.append(filename)

#labelling files in a csv file, in this experiment only 
#two cosmologies were really considered 
with open('labels3.csv','w') as file:  
    for filename in filename_list: 
        #code to extract file number from string of filename 
        number_not_int = filename.partition('-')[-1].rpartition('-')[0]
        a = number_not_int.replace(" ", "")
        try:
            number = int(a)
            if number < 99: 
                l = "1,0"
                file.write(l)
                file.write('\n')
            if number > 98 and number < 197: 
                l = "0,1"
                file.write(l)
                file.write('\n')
            elif number > 198: 
                print(number)
            if number > 196 and number < 295: 
                l = "0,0"
                file.write(l)
                file.write('\n')
            if number > 294 and number < 393: 
                l = "0,0"
                file.write(l)
                file.write('\n')
            if number > 392 and number < 491: 
                l = "0,0"
                file.write(l)
                file.write('\n')
            
        except: 
            pass


#loading the labels that were written to the csv file 
labels = np.genfromtxt("labels3.csv", delimiter=",")
print(len(labels))
print("labels loaded")

#shuffling the data and the labels 
ellips, labels = shuffle(mod_e_list, labels)
print("shuffled!") 

#converting "inf" values to a large number that can 
#be processed in training 
ellips[ellips == inf] = 1000000

#creating a list of plots 
list_of_plots = []
for i in range(386):
    list_of_plots.append((ellips[i])) 


#defining some functions that are used to build the neural net
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 10, 10, 1],
                        strides=[1, 10, 10, 1], padding='VALID')



#creating the variables of the neural network

x = tf.placeholder(tf.float32, shape=[None, 3072*1024])
y_ = tf.placeholder(tf.float32, shape=[None, 2])


x_image = tf.reshape(x, [-1, 3072, 1024, 1])

W_conv1 = weight_variable([40, 40, 1, 5])
b_conv1 = bias_variable([5])

W_conv2 = weight_variable([10, 10, 5, 10])
b_conv2 = bias_variable([10])

W_fc2 = weight_variable([2320, 512])
b_fc2 = bias_variable([512])

W_fc3 = weight_variable([512, 100])
b_fc3 = bias_variable([100])

W_fc4 = weight_variable([100, 20])
b_fc4 = bias_variable([20])

W_fc5 = weight_variable([20, 2])
b_fc5 = bias_variable([2])

W_fc1 = weight_variable([2320, 2320])
b_fc1 = bias_variable([2320])


#forward propagation/network architecture
# with activation functions and dropout

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

h_pool1_flat = tf.reshape(h_pool2, [-1, 2320])
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)
h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)

y_conv = tf.matmul(h_fc4_drop, W_fc5) + b_fc5



#flattening data and labels for input into the CNN
#386 refers to the number of training examples 
plots_flat = [] 
labels_flat = [] 

for i in range(386): 
    plots_flat.append((list_of_plots[i]).flatten())
    labels_flat.append((labels[i]).flatten())
plots_flat_array = np.asarray(plots_flat)
labels_flat_array = np.asarray(labels_flat)

#labelling data as "batches"
batch_x = (plots_flat_array.reshape(386, 3145728)).astype(float)
batch_y = (labels_flat_array.reshape(386, 2)).astype(float)

#loss function  
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
#back propagation 
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#assesing accuracy 
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    count = 0 
    #1000 epochs
    while count < 1000:
        i = randint(0,300)
        print(i)
        count = count + 1 
        #only batches of two images due to RAM limitations
        train_accuracy = accuracy.eval(feed_dict={
                x: batch_x[i:i+2].reshape(2, 3145728),
                      y_: batch_y[i:i+2].reshape(2, 2), keep_prob: 0.5})
        print('step %d, training accuracy %g' % (count, train_accuracy))
        #back prop 
        train_step.run(feed_dict={x: batch_x[i:i+2].reshape(2, 3145728), 
                                     y_: batch_y[i:i+2].reshape(2, 2), keep_prob: 0.5})
