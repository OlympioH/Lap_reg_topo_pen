#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 15:11:51 2021

@author: olympio
"""

#Modules
import numpy               as np
import tensorflow          as tf
import tensorflow_addons   as tfa
import matplotlib.pyplot   as plt
import gudhi               as gd
import math


from difftda               import *
from Test_persistence import chi


from scipy.special import sph_harm
from sklearn import linear_model
from sklearn import kernel_ridge 
from sklearn import manifold
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import kneighbors_graph

#Machines
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')
tf.config.experimental.set_visible_devices([], 'GPU')

def plot3Dmfold(X, Y): #plot a function Y=f(X) as a color map where X 3D array of the points coordinates in the sphere
    fig=plt.figure()
    ax=plt.axes(projection='3d')
    surf=ax.scatter(X[:,0], X[:,1], X[:,2], c=Y)
    fig.colorbar(surf,shrink=0.5, aspect=5)

def chi(st_alpha, f) :#compute the 0 and 1-persistence of a function f on a pre-computed simplicial complex
    st = gd.SimplexTree()
    
    for splx in st_alpha.get_filtration(): #Build a simplicial complex with filtration f.

        if len(splx[0]) == 1:

            st.insert(splx[0],filtration=f[splx[0][0]])

        elif len(splx[0]) == 2:

            st.insert(splx[0],filtration=max(f[splx[0][0]], f[splx[0][1]]))
        elif len(splx[0]) == 3:

            st.insert(splx[0],filtration=max(f[splx[0][0]], f[splx[0][1]], f[splx[0][2]]))
        
            
            
    st.initialize_filtration()

    dgm = st.persistence() #Compute its persistence diagram

    tot_pers0 = 0.0 #Compute its 0 and 1 total persistence
    tot_pers1 = 0.0
    for pt in st.persistence_intervals_in_dimension(0):

        if pt[1] - pt[0] <max_pers:

            tot_pers0 += pt[1] - pt[0]

    for pt in st.persistence_intervals_in_dimension(1):
        if pt[1] - pt[0] <max_pers:

            tot_pers1 += pt[1] - pt[0]
    return np.array([tot_pers0, tot_pers1, dgm]) #Return 0 and 1 persistence and persistence diagram


#%%LB EV regression
#Initialize data on a Swiss roll
g = 1
nb_NN=10
n_pts, nb_comp = 500, 498  
X, t = make_swiss_roll(n_samples = n_pts)
f_true  = 4*np.exp(-((X[:,1]-7)**2/20+(t-6)**2/5))+2*np.cos(t)**2*np.sin(X[:,1])**2
f_true-=np.average(f_true)
sigma=0.2
f= f_true+sigma*np.random.normal(0, 1, n_pts)
f=f-np.average(f)
f=np.array(f, dtype=np.float32)     
        

Adj = kneighbors_graph(X, nb_NN, mode='connectivity') #Compute k-NN graph
X_emb = manifold.spectral_embedding(Adj,n_components=nb_comp,norm_laplacian=True) #Compute EV of graph Laplacian




Alpha_complex=gd.AlphaComplex(X)
st_alpha=Alpha_complex.create_simplex_tree()

file = open("st_alpha.txt", "w") #Simplicial complex on which to compute the persistence
for (s,_) in st_alpha.get_filtration():
    for v in s:
        file.write(str(v) + " ")
    file.write("\n")
file.close()


TP=np.zeros((nb_comp, 1)) #Compute persistence of eigenvectors in 0 and 1 dimension.
for ef in range(nb_comp):
    

    TP[ef, 0] = np.sum(chi(st_alpha, X_emb[:,ef])[0:2])

XTP = np.copy(X_emb)

for i in range(0,nb_comp): #Normliaze columns of design matrix by persistence

#    XP[:,i] = XP[:,i] / Total_pers[i,0]

    if TP[i, 0]< 0.0001:
        XTP[:,i] = XTP[:,i]
    else : 
        XTP[:,i] = XTP[:,i] / (10*TP[i,0])


clf_TP= linear_model.LassoCV(cv=5) #Solve Lasso with penalty Omega_1
clf_TP.fit(XTP, f)
f_TP=clf_TP.predict(XTP)    



idx = np.nonzero(clf_TP.coef_)[0] #Indices selected by Omega_1
X_emb = X_emb [:,idx]
nb_comp = len(idx)
print(nb_comp) 

X_emb=np.array(X_emb, dtype=np.float32)
MSE=[]

#Run penalty Omega 2
thetainit  = np.random.uniform(low=-1., high=1., size=[nb_comp])
theta = tf.Variable(initial_value=np.array(thetainit[:,np.newaxis], dtype=np.float32), trainable=True)
model = SimplexTreeModel(np.array(np.dot(X_emb, thetainit), dtype=np.float32), stbase="st_alpha.txt", dim=[0, 1], card=n_pts//4)
optimizer = tf.keras.optimizers.SGD(learning_rate=4e-2)      
#Gradient descent
losses, dgms, thetas = [], [], []
mu = 3  #Amount of regularization
for epoch in range(800+1):    
    with tf.GradientTape() as tape:
        
        # Compute persistence diagram
        model = SimplexTreeModel(tf.matmul(X_emb, theta),  stbase="st_alpha.txt", dim=[0,1], card=n_pts//4)
        dgm = model.call()

        # Loss is MSE plus the total persistence except for the three most prominent points
        f_estimate=tf.matmul(X_emb, tf.reshape(theta, [-1, 1]))
                
        loss = tf.reduce_sum(tf.square((tf.transpose(f_estimate)-f)) )\
            +mu*tf.reduce_sum(tf.abs(dgm[:,1]-dgm[:,0]))
          
         
            
    # Compute and apply gradients
    gradients = tape.gradient(loss, [theta])
    optimizer.apply_gradients(zip(gradients, [theta]))
    
    losses.append(loss.numpy())
    #dgms.append(dgm)
    thetas.append(theta.numpy()) 
    MSE.append(np.linalg.norm(tf.transpose(tf.matmul(X_emb, theta))-f_true))
    
theta_final=thetas[-1]
f_recons = np.dot (X_emb, theta_final)
  

#%%Plot functions
#plot3Dmfold(X, f_true)
# plt.title('True function')
# plot3Dmfold(X, f)
# plt.title('Noisy observation')
# plot3Dmfold(X, f_recons[:,0])
# plt.title('Topological penalty')     



    
    