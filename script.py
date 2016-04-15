import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys


def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    # IMPLEMENT THIS METHOD
    col_0_sum_mean_list = []
    col_1_sum_mean_list = []
    for i in range (1,6):
        col_0_sum = 0
        col_1_sum = 0
        div = 0
        for j in range(y.shape[0]):
            if y[j] == i:
                col_0_sum = col_0_sum + X[j,0]
                col_1_sum = col_1_sum + X[j,1]
                div = div + 1
        col_0_sum_mean = col_0_sum/div
        col_1_sum_mean = col_1_sum/div
        col_0_sum_mean_list.append(col_0_sum_mean)
        col_1_sum_mean_list.append(col_1_sum_mean)
    means = np.matrix([col_0_sum_mean_list,col_1_sum_mean_list])
    covmat = np.cov(X,rowvar=0)
    return means,covmat


def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    # IMPLEMENT THIS METHOD
    col_0_sum_mean_list = []
    col_1_sum_mean_list = []
    for i in range (1,6):
        col_0_sum = 0
        col_1_sum = 0
        div = 0
        for j in range(y.shape[0]):
            if y[j] == i:
                col_0_sum = col_0_sum + X[j,0]
                col_1_sum = col_1_sum + X[j,1]
                div = div + 1
        col_0_sum_mean = col_0_sum/div
        col_1_sum_mean = col_1_sum/div
        col_0_sum_mean_list.append(col_0_sum_mean)
        col_1_sum_mean_list.append(col_1_sum_mean)
    means = np.matrix([col_0_sum_mean_list,col_1_sum_mean_list])
    covmats = []
    no1 = no2 = no3 = no4 = no5 = 0
    for i in range(y.shape[0]):
        if y[i] == 1:
            no1 += 1
        elif y[i] == 2:
            no2 += 1
        elif y[i] == 3:
            no3 += 1
        elif y[i] == 4:
            no4 += 1
        elif y[i] == 5:
            no5 += 1
    X1 = np.zeros([no1,2])
    X2 = np.zeros([no2,2])
    X3 = np.zeros([no3,2])
    X4 = np.zeros([no4,2])
    X5 = np.zeros([no5,2])
    p = q = r = s = t = 0
    for j in range(y.shape[0]):
        if y[j] == 1:
            X1[p][0] = X[j][0]
            X1[p][1] = X[j][1]
            p += 1
        elif y[j] == 2:
            X2[q][0] = X[j][0]
            X2[q][1] = X[j][1]
            q += 1
        elif y[j] == 3:
            X3[r][0] = X[j][0]
            X3[r][1] = X[j][1]
            r += 1
        elif y[j] == 4:
            X4[s][0] = X[j][0]
            X4[s][1] = X[j][1]
            s += 1
        elif y[j] == 5:
            X5[t][0] = X[j][0]
            X5[t][1] = X[j][1]
            t += 1
    covmat1 = np.cov(X1,rowvar=0)
    covmat2 = np.cov(X2,rowvar=0)
    covmat3 = np.cov(X3,rowvar=0)
    covmat4 = np.cov(X4,rowvar=0)
    covmat5 = np.cov(X5,rowvar=0)
    covmats.append(covmat1)
    covmats.append(covmat2)
    covmats.append(covmat3)
    covmats.append(covmat4)
    covmats.append(covmat5)
    return means,covmats


def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels
    # IMPLEMENT THIS METHOD
    ypred = np.empty([ytest.shape[0],1])
    for i in range(Xtest.shape[0]):
        diff_0 = Xtest[i,0] - means.item(0)
        diff_1 = Xtest[i,1] - means.item(5)
        diff_2 = Xtest[i,0] - means.item(1)
        diff_3 = Xtest[i,1] - means.item(6)
        diff_4 = Xtest[i,0] - means.item(2)
        diff_5 = Xtest[i,1] - means.item(7)
        diff_6 = Xtest[i,0] - means.item(3)
        diff_7 = Xtest[i,1] - means.item(8)
        diff_8 = Xtest[i,0] - means.item(4)
        diff_9 = Xtest[i,1] - means.item(9)
        mat_diff_1 = np.matrix([diff_0,diff_1])
        mat_diff_2 = np.matrix([diff_2,diff_3])
        mat_diff_3 = np.matrix([diff_4,diff_5])
        mat_diff_4 = np.matrix([diff_6,diff_7])
        mat_diff_5 = np.matrix([diff_8,diff_9])
        temp_pred_1=np.dot(np.dot(mat_diff_1,inv(covmat)),mat_diff_1.T)
        temp_pred_2=np.dot(np.dot(mat_diff_2,inv(covmat)),mat_diff_2.T)
        temp_pred_3=np.dot(np.dot(mat_diff_3,inv(covmat)),mat_diff_3.T)
        temp_pred_4=np.dot(np.dot(mat_diff_4,inv(covmat)),mat_diff_4.T)
        temp_pred_5=np.dot(np.dot(mat_diff_5,inv(covmat)),mat_diff_5.T)
        mat_pred = np.matrix([temp_pred_1.item(0),temp_pred_2.item(0),temp_pred_3.item(0),temp_pred_4.item(0),temp_pred_5.item(0)])
        pred_class = (np.argmin(mat_pred) + 1)
        ypred[i][0] = pred_class
    correct = 0
    for i in range(ytest.shape[0]):
        if ytest[i][0] == ypred[i][0]:
            correct += 1
    acc = correct
    return acc,ypred


def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels
    # IMPLEMENT THIS METHOD
    covar_deter_1 = np.linalg.det(covmats[0])
    covar_deter_1 = covar_deter_1**0.5
    covar_deter_1 = covar_deter_1*(2*pi)
    covar_deter_2 = np.linalg.det(covmats[1])
    covar_deter_2 = covar_deter_2**0.5
    covar_deter_2 = covar_deter_2*(2*pi)
    covar_deter_3 = np.linalg.det(covmats[2])
    covar_deter_3 = covar_deter_3**0.5
    covar_deter_3 = covar_deter_3*(2*pi)
    covar_deter_4 = np.linalg.det(covmats[3])
    covar_deter_4 = covar_deter_4**0.5
    covar_deter_4 = covar_deter_4*(2*pi)
    covar_deter_5 = np.linalg.det(covmats[4])
    covar_deter_5 = covar_deter_5**0.5
    covar_deter_5 = covar_deter_5*(2*pi)
    ypred = np.empty([ytest.shape[0],1])
    for i in range(Xtest.shape[0]):
        diff_0 = Xtest[i,0] - means.item(0)
        diff_1 = Xtest[i,1] - means.item(5)
        diff_2 = Xtest[i,0] - means.item(1)
        diff_3 = Xtest[i,1] - means.item(6)
        diff_4 = Xtest[i,0] - means.item(2)
        diff_5 = Xtest[i,1] - means.item(7)
        diff_6 = Xtest[i,0] - means.item(3)
        diff_7 = Xtest[i,1] - means.item(8)
        diff_8 = Xtest[i,0] - means.item(4)
        diff_9 = Xtest[i,1] - means.item(9)
        mat_diff_1 = np.matrix([diff_0,diff_1])
        mat_diff_2 = np.matrix([diff_2,diff_3])
        mat_diff_3 = np.matrix([diff_4,diff_5])
        mat_diff_4 = np.matrix([diff_6,diff_7])
        mat_diff_5 = np.matrix([diff_8,diff_9])
        temp_pred_1=np.exp(0.5*-1*np.dot(np.dot(mat_diff_1,inv(covmats[0])),mat_diff_1.T))/covar_deter_1
        temp_pred_2=np.exp(0.5*-1*np.dot(np.dot(mat_diff_2,inv(covmats[1])),mat_diff_2.T))/covar_deter_2
        temp_pred_3=np.exp(0.5*-1*np.dot(np.dot(mat_diff_3,inv(covmats[2])),mat_diff_3.T))/covar_deter_3
        temp_pred_4=np.exp(0.5*-1*np.dot(np.dot(mat_diff_4,inv(covmats[3])),mat_diff_4.T))/covar_deter_4
        temp_pred_5=np.exp(0.5*-1*np.dot(np.dot(mat_diff_5,inv(covmats[4])),mat_diff_5.T))/covar_deter_5
        mat_pred = np.matrix([temp_pred_1.item(0),temp_pred_2.item(0),temp_pred_3.item(0),temp_pred_4.item(0),temp_pred_5.item(0)])
        pred_class = (np.argmax(mat_pred) + 1)
        ypred[i][0] = pred_class
    correct = 0
    for i in range(ytest.shape[0]):
        if ytest[i][0] == ypred[i][0]:
            correct += 1
    acc = correct
    return acc,ypred


def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD
    w = np.dot(np.dot(inv(np.dot(X.T,X)),X.T),y)
    return w


def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD
    iden = np.identity(X.shape[1])
    w = np.dot(np.dot(inv(lambd*iden + np.dot(X.T,X)),X.T),y)
    return w


def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse
    # IMPLEMENT THIS METHOD
    y_pred = np.dot(Xtest,w)
    error = ytest - y_pred
    error_sq_sum = 0
    for i in range(error.shape[0]):
        error_sq = error[i] * error[i]
        error_sq_sum = error_sq_sum + error_sq
    error_sq_sum = error_sq_sum/error.shape[0]
    rmse = error_sq_sum**0.5
    return rmse


def regressionObjVal(w, X, y, lambd):
    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda
    # IMPLEMENT THIS METHOD
    jw_sum = 0
    for row in range(X.shape[0]):
        jw = (y[row] - np.dot(X[row],w))**2
        jw_sum = jw_sum + jw
    jw_sum = 0.5*jw_sum
    jw_sum = jw_sum + 0.5*lambd*np.dot(w.T,w)
    error = jw_sum
    error_grad = np.empty([w.shape[0],])
    djw_p1 = np.dot(np.dot(X.T,X),w)
    djw_p2 = np.dot(X.T,y)
    djw_p2_vect = np.empty([djw_p2.shape[0],])
    for i in range(djw_p2.shape[0]):
        djw_p2_vect[i] = djw_p2[i][0]
    djw_p3 = lambd*w
    error_grad = djw_p1 - djw_p2_vect + djw_p3
    return (error, error_grad)


def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1))                                                         
    # IMPLEMENT THIS METHOD
    Xd = np.empty([x.shape[0],p+1])
    for i in range(x.shape[0]):
        for j in range(p+1):
            Xd[i][j] = x[i]**j
    return Xd






# Main script

# Problem 1
print "\n\nSTARTING PROBLEM 1..."
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')
# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))
# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()
zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.suptitle('LDA')
plt.show()
zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.suptitle('QDA')
plt.show()
print "\nFINISHED PROBLEM 1!"




# Problem 2
print "\n\n\n\nSTARTING PROBLEM 2..."
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')
# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)
w = learnOLERegression(X,y)
mle_train = testOLERegression(w,X,y)
mle_test = testOLERegression(w,Xtest,ytest)
w_i = learnOLERegression(X_i,y)
mle_train_i = testOLERegression(w_i,X_i,y)
mle_test_i = testOLERegression(w_i,Xtest_i,ytest)
print "\nFor training data:"
print('RMSE without intercept '+str(mle_train))
print('RMSE with intercept '+str(mle_train_i))
print "\nFor test data:"
print('RMSE without intercept '+str(mle_test))
print('RMSE with intercept '+str(mle_test_i))
print "\nFINISHED PROBLEM 2!"




# Problem 3
print "\n\n\n\nSTARTING PROBLEM 3..."
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses3_train = np.zeros((k,1))
rmses3_test = np.zeros((k,1))
for lambd in lambdas:
    #print "\nNow calculating for lambda value",lambd
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3_train[i] = testOLERegression(w_l,X_i,y)
    rmses3_test[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
print "\nFor training data:"
rmses3_train_min = rmses3_train.argmin(0)
print('Minimum value of RMSE with intercept using ridge regression '+str(rmses3_train[rmses3_train_min]))
print "\nFor test data:"
rmses3_test_min = rmses3_test.argmin(0)
print('Minimum value of RMSE with intercept using ridge regression '+str(rmses3_test[rmses3_test_min]))
plt.subplot(211)
plt.plot(lambdas,rmses3_train)
plt.subplot(211)
plt.plot(lambdas,rmses3_test)
print "\nFINISHED PROBLEM 3!"




# Problem 4
print "\n\n\n\nSTARTING PROBLEM 4..."
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args, method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    rmses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.subplot(211)
plt.plot(lambdas,rmses4)
print "\nFINISHED PROBLEM 4!"




# Problem 5
print "\n\n\n\nSTARTING PROBLEM 5..."
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
plt.subplot(212)
plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))
print "\nFINISHED PROBLEM 5!"