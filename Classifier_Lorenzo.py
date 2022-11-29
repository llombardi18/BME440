#!/usr/bin/env python
# coding: utf-8
def runLogisticRegression(Xtrain_raw, Ytrain_raw, Xtest_raw, Ytest_raw, coeffs, ncomponents):
    from sklearn.linear_model import LogisticRegression
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn import metrics

    #print

    #rotate and project along the first ncomponents in pc space
    Xtrain = Xtrain_raw @ coeffs[:, 0:ncomponents]
    Xtest = Xtest_raw @ coeffs[:, 0:ncomponents]
    
    Ytrain = Ytrain_raw
    Ytest = Ytest_raw 

    #set balance
    #print(f"Training set balance: {np.mean(Ytrain)}")
    #print(f"Testing set balance: {np.mean(Ytest)}")

    #normalization?
    normalize = True
    if (normalize == True):
        Ntrain = np.shape(Ytrain)
        Ntest =  np.shape(Ytest)

        centers = np.mean(Xtrain, axis = 0)
        stds = np.std(Xtrain, axis = 0)

        Xtrain =  (Xtrain - centers)/stds
        Xtest = (Xtest - centers)/stds

    #train
    model = LogisticRegression(random_state=0, fit_intercept=1, solver='newton-cg')
    model.fit(Xtrain,Ytrain)
    Ytest_hat = model.predict(Xtest)
    Ytrain_hat = model.predict(Xtrain)

    #decision boundary in the original space
    weights = coeffs[:,0:ncomponents] @ np.transpose(model.coef_)
    #print(weights)

    #plot the scores and the decision boundary in 2D
    print(np.shape(model.coef_))
    print(model.coef_)
    plt.figure()
    for IDclass in range(2):
        idx = (Ytrain == IDclass)
        plt.scatter(Xtrain[idx,0],Xtrain[idx,1], 10, label = IDclass)
    plt.annotate(text='w', xy=(model.coef_[0,0], model.coef_[0,1]), xytext=(0,0), arrowprops=dict(arrowstyle='->'))
    plt.title('Training Set')
    plt.legend()
    plt.xlabel('$PC_1$')
    plt.ylabel('$PC_2$')
    # plt.xlabel('$PC_3$')
    # plt.ylabel('$PC_4$')
    plt.grid()
    plt.show()

    plt.figure()
    for IDclass in range(2):
        idx = (Ytest == IDclass)
        plt.scatter(Xtest[idx,0],Xtest[idx,1], 10, label = IDclass)
    plt.annotate(text='w', xy=(model.coef_[0,0], model.coef_[0,1]), xytext=(0,0), arrowprops=dict(arrowstyle='->'))
    plt.title('Test Set')
    plt.legend()
    plt.xlabel('$PC_1$')
    plt.ylabel('$PC_2$')
    # plt.xlabel('$PC_3$')
    # plt.ylabel('$PC_4$')
    plt.grid()
    plt.show()

    #testing: confusion matrix
    plt.figure()
    cm = metrics.confusion_matrix(Ytest,Ytest_hat)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm)
    cm_display.plot()
    plt.show()

    print(f" f1: {metrics.f1_score(Ytest,Ytest_hat)}")
    fpr, tpr, thresholds = metrics.roc_curve(Ytest, Ytest_hat, pos_label=2)
    print(f" auc: {metrics.auc(fpr, tpr)}")

    #In sample vs out of sample error
    Ein = np.mean(np.round(Ytrain) != np.round(Ytrain_hat))
    Eout = np.mean(np.round(Ytest) != np.round(Ytest_hat))
    print(f"In sample error: {Ein}")
    print(f"Out of sample error: {Eout}")

    return Ein, Eout

def test():
    print("Hello")

