#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 15:32:32 2021

@author: olympio
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:31:13 2020

@author: olympioooo
"""
#Modules
import numpy               as np
import random as rd
import tensorflow          as tf
import tensorflow_addons   as tfa
import matplotlib.pyplot   as plt
import gudhi               as gd
from scipy import ndimage


from difftda               import *
from skimage import color
from skimage import io
from sklearn import kernel_ridge 
from sklearn import linear_model
from sklearn import manifold
from scipy.spatial.distance import cdist
from sklearn.neighbors import kneighbors_graph

#Machines
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')
tf.config.experimental.set_visible_devices([], 'GPU')

def sigmo(x):
    return 1/(1+np.exp(-x))

#%% Building dataset
    
   

#t=100 ?
Y=[]
X=np.zeros((72, 72, 16384)) #Create a dataset of 72*72 images rotated (object + image)
for i in range(72):
    Im0=color.rgb2gray(io.imread('data/cat__%s.png' %(5*i)))
    for j in range(72):
        Im = ndimage.rotate(Im0, 5*j, reshape = False)
        Im = Im.flatten()
        X[i, j] = Im 
n_pts = 1000
nb_comp = 900
X_sub=[]
Y = np.zeros(n_pts)
for N in range(n_pts):  #Sample 1000 images amongst the whole dataset
    i = np.random.randint(72)
    j = np.random.randint(72)
    i_deg, j_deg = 5*i, 5*j #Convert the integers between 0 and 71 into an angle in degrees between 0 and 355
    theta, phi = np.pi*i_deg/180, np.pi*j_deg/180 #Convert the angles from degrees to rad
    X_sub.append(X[i, j])  
    Y[N]=sigmo(-17*(np.sqrt((theta-np.pi)**2+(phi-np.pi)**2)-0.6*np.pi))
Y=Y-np.average(Y)
X= X_sub
X=np.array(X)

    
    




#%%
#Split Train and test set

pert=1
split_train_test=True
    
X=X[:n_pts]
size_train=900
size_test=100
train_indices=rd.sample(range(n_pts), size_train)
train_indices = np.sort(train_indices)
test_indices=np.delete(np.arange(n_pts), train_indices)
test_indices = np.sort (test_indices)
#train_indices=np.arange(68)
#test_indices=[68, 69, 70, 71]
X_train=X[train_indices]
X_test=X[test_indices]
noise=np.random.normal(0, pert, n_pts)
Y_train=Y[train_indices]+noise[train_indices]
Y_test_true=Y[test_indices]
Y_test=Y[test_indices]+noise[test_indices]

    

    
#Build graph on the data
w_graph=False #Gaussian graph
kNN=True #K-NN graph

if w_graph:

    t=10 #Scaling parameter of the graph
    d=cdist(X,X)
    Adj=np.exp(-d**2/(4*t))
    for i in range(n_pts):
        Adj[i,i]=0
    st_alpha=gd.SimplexTree()
    thresh=10
    for  i in range(n_pts):
        for j in range(i):
            if d[i,j]<thresh:
                st_alpha.insert([i,j])
        


if kNN:
    nb_NN = 10
    
    Adj = kneighbors_graph(X, nb_NN, mode='connectivity')
    
    st_alpha = gd.SimplexTree()
    
    for i in range(n_pts):
    
        for j in range(n_pts):
    
            if Adj[i,j] != 0:
    
                st_alpha.insert([i,j])
print('graph computed !')

X_emb = manifold.spectral_embedding(Adj,n_components=nb_comp,norm_laplacian=False) #Compute eigenbasis
X_emb=np.array(X_emb, dtype=np.float32)
print('embedding done!')
X_emb_train=X_emb[train_indices]
X_emb_test=X_emb[test_indices]

Total_pers0 = np.zeros((nb_comp,1))
Total_pers1 = np.zeros((nb_comp,1))
max_pers = np.inf


for ef in range(nb_comp): #Compute the persistence on a simplicial complex built on the data

    st = gd.SimplexTree()

    for splx in st_alpha.get_filtration():

        if len(splx[0]) == 1:
                    
            st.insert(splx[0],filtration=X_emb[splx[0][0],ef])
            

        elif len(splx[0]) == 2:

            st.insert(splx[0],filtration=max(X_emb[splx[0][0],ef],X_emb[splx[0][1],ef]))
    st.expansion(2)
            
    st.initialize_filtration()

    dgm = st.persistence()

    tot_pers0 = 0.0
    tot_pers1 = 0.0
    for pt in st.persistence_intervals_in_dimension(0):

        if pt[1] - pt[0] <max_pers:

            tot_pers0 += pt[1] - pt[0]

#           tot_pers += np.sqrt(pt[1] - pt[0])
    for pt in st.persistence_intervals_in_dimension(1):
        if pt[1] - pt[0] <max_pers:

            tot_pers1 += pt[1] - pt[0]
        
    Total_pers0[ef,0] = tot_pers0
    Total_pers1[ef,0] = tot_pers1
    Total_pers = Total_pers0 + Total_pers1 

XP = np.copy(X_emb)


for i in range(0,nb_comp):

#    XP[:,i] = XP[:,i] / Total_pers[i,0]

    if Total_pers[i] <0.0001:
        XP[:,i] = XP[:,i]

    else:

        XP[:,i] = XP[:,i] / (10*Total_pers[i])
        
 
XP_train=XP[train_indices]
XP_test=XP[test_indices]
 
clf0 = linear_model.LassoCV()
clf0.fit(XP_train,Y_train)
Y_TP= clf0.predict(XP_test)
        

print("----------------------------------------------")

print("Number of coeffs for Omega1 penalty: ", np.sum(clf0.coef_ != 0.0))

print("MSE on Omega 1 penalty: ", np.linalg.norm(Y_TP-Y_test_true))

print("----------------------------------------------")


#%%

file = open("st_rotation.txt", "w") #Simplicial complex on which to compute the persistence
for (s,_) in st.get_filtration():
    for v in s:
        file.write(str(v) + " ")
    file.write("\n")
file.close()

idx = np.nonzero(clf0.coef_)[0] #Restrict to eigenvectors selected by Omega1
X_emb = X_emb [:,idx]
XP=XP[:,idx]
XP_train = XP_train [:,idx]
XP_test = XP_test[:,idx]
nb_comp = len(idx)
MSE=[]
#Variables, optimizer
betainit  = np.random.uniform(low=-1., high=1., size=[nb_comp])
beta = tf.Variable(initial_value=np.array(betainit[:,np.newaxis], dtype=np.float32), trainable=True)
model = SimplexTreeModel(np.array(np.dot(XP_train, betainit), dtype=np.float32), stbase="st_rotation.txt", dim=[0], card=n_pts//4)
optimizer = tf.keras.optimizers.SGD(learning_rate=4e-6)      
#Gradient descent
losses, dgms, betas = [], [], []
mu = 5 #Amount of regularization
for epoch in range(2000+1):    
    with tf.GradientTape() as tape:
        
        # Compute persistence diagram
        model = SimplexTreeModel(tf.matmul(XP, beta),  stbase="st_rotation.txt", dim=[0], card=n_pts//4)
        dgm = model.call()

        # Loss is MSE plus the total persistence except for the three most prominent points
        f_estimate=tf.matmul(XP_train, tf.reshape(beta, [-1, 1]))
                
        loss = tf.reduce_sum(tf.square((tf.transpose(f_estimate)-Y_train)) )\
            +mu*tf.reduce_sum(tf.abs(dgm[:,1]-dgm[:,0]))
          
         
            
    # Compute and apply gradients
    gradients = tape.gradient(loss, [beta])
    optimizer.apply_gradients(zip(gradients, [beta]))
    
    losses.append(loss.numpy())
    #dgms.append(dgm)
    betas.append(beta.numpy())
    
    MSE.append(np.linalg.norm(tf.transpose(tf.matmul(XP_test, beta))-Y_test_true))   
beta_final=betas[-1]
f_recons = np.dot (XP_test, beta_final)

print("MSE on topo-pen reg: ", np.linalg.norm(tf.transpose(f_recons)-Y_test_true))

    
    


