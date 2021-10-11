import math

import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys


def ldaLearn(X, y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix
    # print(np.concatenate((X,y),axis=1))
    # print("-----------------Conditioning-----------------")
    # print(np.mean(X[np.where(y==2)[0]]))
    covmat = np.cov(np.transpose(X))
    # Combine lables and samples
    totalMatrix = np.concatenate((X, y), axis=1)

    # by class
    means = []
    for i in np.unique(y):
        data = totalMatrix[np.where(totalMatrix[..., 2] == i)][:, [0, 1]]
        mean = np.mean(data, axis=0)
        means.append(mean)
    means = np.transpose(means)

    return means, covmat


def qdaLearn(X, y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes

    # Combine lables and samples
    totalMatrix = np.concatenate((X, y), axis=1)

    # by class
    means = []
    covmats = []
    for i in np.unique(y):
        data = totalMatrix[np.where(totalMatrix[..., 2] == i)][:, [0, 1]]
        mean = np.mean(data, axis=0)
        covmats.append(np.cov(np.transpose(data), bias=1))
        means.append(mean)
    means = np.transpose(means)

    return means, covmats


def ldaTest(means, covmat, Xtest, ytest):
    # Inputs
    # means d x K, covmat d x d - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    ypred = np.empty([Xtest.shape[0], 1])
    for sample in range(Xtest.shape[0]):
        predict = 0
        i = 0
        for index in range(means.shape[1]):
            p = np.add(
                np.log(pi),
                np.subtract(
                    np.dot(np.dot(np.transpose(Xtest[sample]), inv(covmat)), means[:, index]),
                    np.multiply(0.5, np.dot(np.dot(np.transpose(means[:, index]), inv(covmat)), means[:, index]))
                )
            )
            if p > predict:
                predict = p
                i = index
        ypred[sample, 0] = (i + 1)

    accurate = 0
    for i in range(ypred.shape[0]):
        if ypred[i][0] == ytest[i][0]:
            accurate += 1
    return accurate, ypred


def qdaTest(means, covmats, Xtest, ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    ypred = np.empty([Xtest.shape[0], 1])
    d = Xtest.shape[1]
    for sample in range(Xtest.shape[0]):
        predict = 0
        i = 0
        for index in range(means.shape[1]):
            subtract = np.subtract(Xtest[sample], means[:, index])
            p = np.divide(
                np.exp(
                    np.multiply(-0.5,
                                np.dot(subtract,
                                       np.dot(np.transpose(
                                           subtract), inv(covmats[index])
                                       )))),
                np.multiply(np.sqrt(det(covmats[index])), np.power(2 * pi, math.floor(d / 2)))
            )
            if np.max(p) > predict:
                predict = np.max(p)
                i = index
        ypred[sample, 0] = (i + 1)

    correct = 0
    for i in range(ypred.shape[0]):
        if ypred[i][0] == ytest[i][0]:
            correct = correct + 1
    acc = correct
    return acc, ypred
    return acc, ypred


def learnOLERegression(X, y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1
    # IMPLEMENT THIS METHOD

    w = np.dot(np.dot(inv(np.dot(np.transpose(X), X)), np.transpose(X)), y)

    return w


def learnRidgeRegression(X, y, lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    I = np.identity(X.shape[1])

    w = np.dot(np.dot(np.linalg.inv(np.dot(lambd, I) + np.dot(np.transpose(X), X)), np.transpose(X)), y)

    # IMPLEMENT THIS METHOD                                                   
    return w


def testOLERegression(w, Xtest, ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    N = Xtest.shape[0]
    mse = np.sum(np.power(np.subtract(ytest, np.dot(Xtest, w)), 2))
    mse = np.divide(mse, N)
    # IMPLEMENT THIS METHOD
    return mse


def regressionObjVal(w, X, y, lambd):
    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD
    new_w = w.reshape(65, 1)
    # error = 0.5*((y - w*X).T * (y - W)) + 0.5*lambd(w.T*w)
    e_part1 = np.subtract(y, np.dot(X,new_w))
    e_part2 = np.multiply(0.5,np.multiply(lambd,np.dot(np.transpose(new_w),new_w)))

    error = np.add(
        np.multiply(0.5, np.dot(
            np.transpose(e_part1),e_part1)),
        e_part2)
    # error_grad = (X.T*X)W - X.T*y + lambd*w
    eg_part1 = np.dot(np.dot(np.transpose(X),X), new_w)
    eg_part2 = np.dot(np.transpose(X),y)
    eg_part3 = np.multiply(lambd, new_w)

    error_grad = np.add(np.subtract(eg_part1,eg_part2),eg_part3)

    return error.flatten(), error_grad.flatten()


def mapNonLinear(x, p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 

    Xp = np.ones((x.shape[0],p+1)) #N = rows in x
    for i in range(1,p+1):
        Xp[:,i] = math.pow(x,i)
    return Xp


# Main script
np.set_printoptions(suppress=True)
# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X, y, Xtest, ytest = pickle.load(open('sample.pickle', 'rb'))
else:
    X, y, Xtest, ytest = pickle.load(open('sample.pickle', 'rb'), encoding='latin1')

# LDA
means, covmat = ldaLearn(X, y)
# print(means)
# print (covmat)
ldaacc, ldares = ldaTest(means, covmat, Xtest, ytest)
print('LDA Accuracy = ' + str(ldaacc))
# # QDA
means, covmats = qdaLearn(X, y)
qdaacc, qdares = qdaTest(means, covmats, Xtest, ytest)
print('QDA Accuracy = ' + str(qdaacc))
#
# plotting boundaries
x1 = np.linspace(-5, 20, 100)
x2 = np.linspace(-5, 20, 100)
xx1, xx2 = np.meshgrid(x1, x2)
xx = np.zeros((x1.shape[0] * x2.shape[0], 2))
xx[:, 0] = xx1.ravel()
xx[:, 1] = xx2.ravel()

fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)

zacc, zldares = ldaTest(means, covmat, xx, np.zeros((xx.shape[0], 1)))
plt.contourf(x1, x2, zldares.reshape((x1.shape[0], x2.shape[0])), alpha=0.3)
plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest.ravel())
plt.title('LDA')
#
plt.subplot(1, 2, 2)

zacc, zqdares = qdaTest(means, covmats, xx, np.zeros((xx.shape[0], 1)))
plt.contourf(x1, x2, zqdares.reshape((x1.shape[0], x2.shape[0])), alpha=0.3)
plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest.ravel())
plt.title('QDA')
#
plt.show()
# Problem 2
if sys.version_info.major == 2:
    X, y, Xtest, ytest = pickle.load(open('diabetes.pickle', 'rb'))
else:
    X, y, Xtest, ytest = pickle.load(open('diabetes.pickle', 'rb'), encoding='latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0], 1)), Xtest), axis=1)

w = learnOLERegression(X, y)
mle = testOLERegression(w, Xtest, ytest)

w_i = learnOLERegression(X_i, y)
mle_i = testOLERegression(w_i, Xtest_i, ytest)
print('MSE without intercept ' + str(mle))
print('MSE with intercept ' + str(mle_i))




# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
optimal = sys.maxsize
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    w_ole = learnOLERegression(X_i,y)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    if (mses3[i] < optimal):
        w_ole_op = w_ole
        optimal = mses3[i]
        l = lambd
        w_ridge = w_l
    i = i + 1
print('Optimal lambda '+str(l))
print('MSE Ridge with intercept '+str(optimal))
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()





# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt = 0 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
