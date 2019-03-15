#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 04:37:24 2017

@author: Andres Vargas
"""


from __future__ import division
import numpy as np
import tensorflow as tf
import os
import time


def loadBatch(currperm,memmap):
    return {'left_eye':memmap['train_eye_left'][currperm,:,:,:]/255,\
           'right_eye':memmap['train_eye_right'][currperm,:,:,:]/255,\
           'face':memmap['train_face'][currperm,:,:,:]/255,\
           'mask':memmap['train_face_mask'][currperm,:,:],\
           'y':memmap['train_y'][currperm,:]}

def loadVal(currperm,memmap):
    return {'left_eye':memmap['val_eye_left'][currperm,:,:,:]/255,\
           'right_eye':memmap['val_eye_right'][currperm,:,:,:]/255,\
           'face':memmap['val_face'][currperm,:,:,:]/255,\
           'mask':memmap['val_face_mask'][currperm,:,:],\
           'y':memmap['val_y'][currperm,:]}



# the size of the inputs should be as follows:
    #num_layers: list of size 3: first dimension corresponds to eyes, 2nd to face, 3rd to mask
    #num_filters: nested list; outer dimension size 3, ith inner dimension has size equal to num_layers[i] 
    #filter sizes:nested list; outer dimension size 3; ith inner dimension has size equal to num_layers[i]
def getLayerSizes(num_layers,filter_sizes,filter_strides,mask_layer_sizes):
    
    eye_layer_sizes = [64] #start of with 64x64 eye image
    for i in range(num_layers[0]):# e stand for "eyes"
        eye_layer_sizes.append((eye_layer_sizes[i]-filter_sizes[0][i])/filter_strides[0][i] + 1)
        
    face_layer_sizes = [64] #start of with 64x64 face image
    for i in range(num_layers[1]):# e stand for "eyes"
        face_layer_sizes.append((face_layer_sizes[i]-filter_sizes[1][i])/filter_strides[1][i] + 1)
        
    mask_layer_sizes_flat = [25**2]
    for i in range(num_layers[2]):
        mask_layer_sizes_flat.append(mask_layer_sizes[i])
                
    return [eye_layer_sizes,face_layer_sizes,mask_layer_sizes_flat]
    
def conv(input_,num_filts,kernel_size,strides,pad,name):
    return tf.layers.conv2d(
            inputs=input_,
            filters=num_filts,
            kernel_size=kernel_size,
            strides=strides,
            padding=pad,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(uniform=False,factor=0.1),
            name=name)
    
def pool(input_,size,strides,name): 
    return tf.layers.max_pooling2d(input_,pool_size=size,strides=strides,name=name)

def dense(input_,out_dim,activ,name):
	return tf.layers.dense(input_,units=out_dim,
		kernel_initializer=tf.contrib.layers.variance_scaling_initializer(uniform=False,factor=0.1),
		activation = activ,
		name=name)

def shapelist(tens):
	return tens.get_shape().as_list()
     
###  BEGIN IMPLEMENTATION
os.chdir('DIRECTORY_CONTAINING_train_and_val_dot_npz')
summary_path = 'TARGET_DIRETORY_FOR_TENSORBOARD_VISUALIZATIONS'
npzfile = np.load("train_and_val.npz",mmap_mode='r')
bsize =  500 #batch size
val_size = 3000 #size of validation set 
train_size = 5000 #only use two thirds of training data to compute training error, to save computation time
N = 48000
Ntest = 5000

num_layers = [3,2]
filter_sizes = [[5,3,3],[5,3]] 

#for the eyes, after the first convolution, we have 60x60, 2x2 pool to get 30x30.  
#Then convolve again to get 28x28.  2x2 max pool to get 14x14. convolve to get 12x12.  2x2 max pool to get 6x6
# same process for face, except stop after second max pool, so we have 14x14
filter_strides = [[1,1,1],[1,1]]#
mask_layer_sizes = [625,100]

#out = getLayerSizes(num_layers,filter_sizes,filter_strides,mask_layer_sizes) 
# first element of out is size of each eye output
# second element of out is size of face output


num_filt_CF0 = 18 
num_filt_CF1 = 36

num_filt_CE0 = 18
num_filt_CE1 = 36
num_filt_CE2 = 72 

size_CE0 = (filter_sizes[0][0],filter_sizes[0][0])
size_CE1 = (filter_sizes[0][1],filter_sizes[0][1])
size_CE2 = (filter_sizes[0][2],filter_sizes[0][2])

size_CF0 = (filter_sizes[1][0],filter_sizes[1][0])
size_CF1 = (filter_sizes[1][1],filter_sizes[1][1])

strides_CE0 = (filter_strides[0][0],filter_strides[0][0])
strides_CE1 = (filter_strides[0][1],filter_strides[0][1])
strides_CE2 = (filter_strides[0][2],filter_strides[0][2])

strides_CF0 = (filter_strides[1][0],filter_strides[1][0])
strides_CF1 = (filter_strides[1][1],filter_strides[1][1])


#iterate through some hyperparameters to find the best setting
rate = 0.001
lambd = 0.1

print 'parameters: rate-', str(rate), 'lambda-', str(lambd)
tf.reset_default_graph()

left_eye = tf.placeholder(tf.float32,shape=(None,64,64,3),name='left_eye')
right_eye = tf.placeholder(tf.float32,shape=(None,64,64,3),name='right_eye')
face = tf.placeholder(tf.float32,shape=(None,64,64,3),name='face')
mask = tf.placeholder(tf.float32,shape=(None,25,25),name='mask')
y = tf.placeholder(tf.float32,shape=(None,2),name='y')

## EYES
CRE0 = conv(right_eye,num_filt_CE0,size_CE0,strides_CE0,'same','CRE0')
PRE0 = pool(CRE0,2,2,'PRE0')
CRE1 = conv(PRE0,num_filt_CE1,size_CE1,strides_CE1,'same','CRE1')
PRE1 = pool(CRE1,2,2,'PRE1')
CRE2 = conv(PRE1,num_filt_CE2,size_CE2,strides_CE2,'same','CRE2')
PRE2 = pool(CRE2,2,2,'PRE2')
FRE = tf.reshape(PRE2,\
                [-1]+ [shapelist(PRE2)[1]*shapelist(PRE2)[2]*shapelist(PRE2)[3]] \
           ,name='FRE') 
 
CLE0 = conv(left_eye,num_filt_CE0,size_CE0,strides_CE0,'same','CLE0')
PLE0 = pool(CLE0,2,2,'PLE0')
CLE1 = conv(PLE0,num_filt_CE1,size_CE1,strides_CE1,'same','CLE1')
PLE1 = pool(CLE1,2,2,'PLE1')
CLE2 = conv(PLE1,num_filt_CE2,size_CE2,strides_CE2,'same','CLE2')
PLE2 = pool(CLE2,2,2,'PLE2')
FLE = tf.reshape(PLE2,\
		[-1]+ [shapelist(PLE2)[1]*shapelist(PLE2)[2]*shapelist(PLE2)[3]] \
           ,name='FLE')
FE = tf.concat([FRE,FLE],axis=1,name='FE') 

#couple of fully connected layers for the eyes
DE0 = dense(FE,1000,tf.nn.relu,'DE0')
DE1 = dense(DE0,1000,tf.nn.relu,'DE1')


## FACE
CF0 = conv(face,num_filt_CF0,size_CF0,strides_CF0,'same','CF0')
PF0 = pool(CF0,2,2,'PF0')
CF1 = conv(PF0,num_filt_CF1,size_CF1,strides_CF1,'same','CF1')
PF1 = pool(CF1,2,2,'PF1')
FF = tf.reshape(PF1,\
		[-1,shapelist(PF1)[1]*shapelist(PF1)[2]*num_filt_CF1],\
		name='FF')

# couple of fully connected layers for face
DF0 = dense(FF,1000,tf.nn.relu,'DF0')
DF1 = dense(DF0,1000,tf.nn.relu,'DF1')


## flatten mask and add a few dense layers for it
FM = tf.transpose(tf.reshape(mask,[25*25,-1],name='FM'))
DM0 = dense(FM,625,tf.nn.relu,'DM0')
DM1 = dense(FM,625,tf.nn.relu,'DM1')

#outputs
final_flat = tf.nn.dropout(dense(tf.concat([DE1,DF1,DM1],axis=1),1000,tf.nn.relu,None),keep_prob=0.5,name='final_flat')
pred = dense(final_flat,2,None,'pred') 
err = tf.reduce_mean(tf.norm(pred-y,axis=1),name='error')
weights = tf.trainable_variables() # all vars of your graph
regularized_loss = err
step = tf.train.AdamOptimizer(learning_rate=rate).minimize(err)

#summaries
trainerrwriter = tf.summary.FileWriter(summary_path+'/trainerr_rate'+str(rate)+'lambda'+str(lambd))
testerrwriter = tf.summary.FileWriter(summary_path+'/testerr_rate'+str(rate)+'lambda'+str(lambd))
trainlosswriter = tf.summary.FileWriter(summary_path+'/trainloss_rate'+str(rate)+'lambda'+str(lambd))
testlosswriter =  tf.summary.FileWriter(summary_path+'/testloss_rate'+str(rate)+'lambda'+str(lambd))
 
error_summary = tf.summary.scalar('error',err)
loss_summary = tf.summary.scalar('loss',regularized_loss)

#Model Saving
tf.get_collection("validation_nodes")
tf.add_to_collection("validation_nodes", left_eye)
tf.add_to_collection("validation_nodes", right_eye)
tf.add_to_collection("validation_nodes", face)
tf.add_to_collection("validation_nodes", mask)
tf.add_to_collection("validation_nodes", pred)
saver = tf.train.Saver()

np.random.seed(12300)
seq = range(N)
seq_val = range(Ntest)
start = time.time()
val = loadVal(np.random.permutation(seq_val)[:val_size],npzfile)       
train  = loadBatch(np.random.permutation(seq)[:train_size],npzfile)
valerr = 20
with tf.Session() as session:
	tf.global_variables_initializer().run()
	trainerrwriter.add_graph(session.graph)
	cnt = 0
	abscnt = 0
	epoch = 0
	perm = np.random.permutation(seq)
	while epoch<25: 
	    print abscnt
	    currstart = cnt*bsize
	    currperm = perm[currstart:currstart+bsize]
	    batch = loadBatch(currperm,npzfile)
	    session.run(step,{left_eye:batch['left_eye'],right_eye:batch['right_eye'],\
			face:batch['face'],mask:batch['mask'],y:batch['y']})
        if abscnt % 47 == 0: # to save time, only compute errors every 47 iterations.  47 chosen arbitrarily 
    	    s1 = session.run(error_summary,{left_eye:train['left_eye'],right_eye:train['right_eye'],\
    							       face:train['face'],mask:train['mask'],y:train['y']})
    	    trainerr = session.run(err,{left_eye:train['left_eye'],right_eye:train['right_eye'],\
    							       face:train['face'],mask:train['mask'],y:train['y']})  
    	    trainloss =  session.run(loss_summary,{left_eye:train['left_eye'],right_eye:train['right_eye'],\
    							       face:train['face'],mask:train['mask'],y:train['y']})  
    
    	    print trainerr
            trainerrwriter.add_summary(s1,abscnt)
    	    trainlosswriter.add_summary(trainloss,abscnt)
    
    	    s2 = session.run(error_summary,{left_eye:val['left_eye'],right_eye:val['right_eye'],\
    							       face:val['face'],mask:val['mask'],y:val['y']})  
    	    valerr =  session.run(err,{left_eye:val['left_eye'],right_eye:val['right_eye'],\
    							       face:val['face'],mask:val['mask'],y:val['y']})
    	    valloss =  session.run(loss_summary,{left_eye:val['left_eye'],right_eye:val['right_eye'],\
    							       face:val['face'],mask:val['mask'],y:val['y']})
    	    print valerr 
    	    testerrwriter.add_summary(s2,abscnt)
    	    testlosswriter.add_summary(valloss,abscnt)
	    cnt += 1
	    abscnt += 1
	    if (cnt*bsize+bsize)>(N-1):
		epoch += 1
		print epoch, 'epochs elapsed'
		cnt = 0
		perm = np.random.permutation(seq) 
	    save_path = saver.save(session, os.getcwd()+"/my_model.ckpt")
	    end = time.time()-start   
	print end     

