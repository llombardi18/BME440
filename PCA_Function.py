#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# PCA function 
def principalCA(X_training_data):
    import numpy as np
    from matplotlib import pyplot as plt

   # calculate the  covariance 
    C = np.cov(X_training_data.transpose())
    plt.matshow(C)
    print(C.shape)

    # Calcualte the eigenvalue and eigenvector of covariance matrix 
    eigval, eigvec = np.linalg.eig(C)
    print("Eigenvalue shape: ",eigval.shape)
    print("Eigenvector shape: ",eigvec.shape) 
    return eigval, eigvec

