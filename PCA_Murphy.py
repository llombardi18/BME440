#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
from matplotlib import pyplot as plt
from numpy import loadtxt
dataset = loadtxt('DatasetEEGProject.txt', dtype=float, delimiter=",", unpack=False)
print(dataset.shape)


from sklearn.model_selection import train_test_split
X_training_data,X_test_data,Y_training_data,Y_test_data =train_test_split(dataset[:,:64],dataset[:,64],test_size= 0.3,random_state=0)
print(X_training_data.shape)
print(X_test_data.shape)


# In[2]:


plt.plot(X_training_data,'.')
plt.xlabel("Subject Number")
plt.ylabel("Avergae beta power accross trials")


# In[26]:


# PCA function 
def principalCA(X_training_data):
   # calculate the  covariance 
    C = np.cov(X_training_data.transpose())
    plt.matshow(C)
    print(C.shape)

    # Calcualte the eigenvalue and eigenvector of covariance matrix 
    eigval, eigvec = np.linalg.eig(C)
    print("Eigenvalue shape: ",eigval.shape)
    print("Eigenvector shape: ",eigvec.shape) 
    return eigval, eigvec
    


# In[5]:


plt.plot(eigval/sum(eigval),'.')


# In[15]:


# PCA for first two component 
Xpca = np.dot(X_training_data,eigvec[:,0:5])
print(X_training_data.shape)
print(Xpca.shape)


# In[14]:


plt.scatter(Xpca[:,0],Xpca[:,1],10,Y_training_data)
plt.legend(['Left','Right'])
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')


# In[22]:


# plt.figure(figsize=(6,5))
# for i in range (5):
#     plt.scatter(Xpca[i,0],Xpca[i,1],cmap="plasma",label = i)
#     plt.legend()
#     plt.show


# In[24]:


plt.imshow(eigvec[:,0:5],aspect=0.1)
plt.colorbar()


# In[ ]:





# In[ ]:




