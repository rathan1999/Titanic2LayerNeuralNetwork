import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import *
def sigmoid(z):
    return 1/(1+np.exp(-1*z))
def normalise(Xtrain):
    return (Xtrain - np.mean(Xtrain,axis=1,keepdims=True))/(np.max(Xtrain,axis=1,keepdims=True)-np.min(Xtrain,axis=1,keepdims=True))

def load_data():
    data=pd.read_csv("C:\\Users\\Sai Rathan\\Desktop\\train.csv")
    data['Sex'].replace(['female','male'],[1,2],inplace=True)
    data['Age'].replace(np.NaN,data['Age'].mean(),inplace=True)
    data['Pclass'].replace(np.NaN,data['Pclass'].mean(),inplace=True)
    data['Sex'].replace(np.NaN,data['Sex'].mean(),inplace=True)
    data['SibSp'].replace(np.NaN,data['SibSp'].mean(),inplace=True)
    data['Parch'].replace(np.NaN,data['Parch'].mean(),inplace=True)
    data['Fare'].replace(np.NaN,data['Fare'].mean(),inplace=True)
    data['Survived'].replace(np.NaN,data['Survived'].mean(),inplace=True)
    data['Embarked'].replace(['C','S','Q'],[1,2,3],inplace=True)
    data['Embarked'].replace(np.NaN,data['Embarked'].mean(),inplace=True)
    Datatrain=data.sample(frac=0.9,random_state=200)
    Dataval=data.drop(Datatrain.index)
    Xtrain=np.array(Datatrain[['Pclass','Sex','Age','Fare','SibSp','Parch','Embarked']]).T
    Xtrain = normalise(Xtrain)
    Xval=np.array(Dataval[['Pclass','Sex','Age','Fare','SibSp','Parch','Embarked']]).T
    Xval=normalise(Xval)
    Ytrain=np.array(Datatrain['Survived']).reshape(1,Xtrain.shape[1])
    Yval=np.array(Dataval['Survived']).reshape(1,Xval.shape[1])
    return [Xtrain,Ytrain,Xval,Yval]

def initialize():
    W=[]
    L1=5
    L2=1
    alpha=0.08
    W.append(np.random.randn(L1,Xtrain.shape[0])*0.05)
    W.append(np.random.randn(L2,L1)*0.05)
    B=[]
    B.append(np.random.randn(L1,1)*0.05)
    B.append(np.random.randn(L2,1)*0.05)
    return L1,L2,W,B,alpha

def trainonce(Xtrain,Ytrain,W,B,alpha):
    Z=[Xtrain]
    A=[Xtrain]
    Z.append(np.dot(W[0],Xtrain)+B[0])
    A.append(sigmoid(Z[-1]))
    Z.append(np.dot(W[1],A[-1])+B[1])
    A.append(sigmoid(Z[-1]))
    J=np.sum(-1*(Ytrain*np.log(A[-1])+(1-Ytrain)*np.log(1-A[-1])),axis=1,keepdims=True)/Xtrain.shape[1]
    dz2=A[-1]-Ytrain
    dw2=np.dot(dz2,(A[-2].T))/Xtrain.shape[1]
    db2=np.sum(dz2,axis=1,keepdims=True)/Xtrain.shape[1]
    dz1=np.dot(W[1].T,dz2)*(1-A[1])
    dw1=np.dot(dz1,Xtrain.T)/Xtrain.shape[1]
    db1=np.sum(dz1,axis=1,keepdims=True)/Xtrain.shape[1]
    W[0]=W[0]-alpha*dw1
    W[1]=W[1]-alpha*dw2
    B[0]=B[0]-alpha*db1
    B[1]=B[1]-alpha*db2
    return J

def crossvalidate(Xval,Yval,W,B):
    Z=[Xval]
    A=[Xval]
    Z.append(np.dot(W[0],Xval)+B[0])
    A.append(sigmoid(Z[-1]))
    Z.append(np.dot(W[1],A[-1])+B[1])
    A.append(sigmoid(Z[-1]))
    J=np.sum(-1*(Yval*np.log(A[-1])+(1-Yval)*np.log(1-A[-1])),axis=1,keepdims=True)/Xval.shape[1]
    print(J)
    x=np.around(A[-1],0)-Yval
    err=np.sum(x*x)
    print(((Xval.shape[1]-err)*100)/Xval.shape[1])

def load_datatest():
    data=pd.read_csv("C:\\Users\\Sai Rathan\\Desktop\\test.csv")
    data['Sex'].replace(['female','male'],[1,2],inplace=True)
    data['Age'].replace(np.NaN,data['Age'].mean(),inplace=True)
    data['Pclass'].replace(np.NaN,data['Pclass'].mean(),inplace=True)
    data['Sex'].replace(np.NaN,data['Sex'].mean(),inplace=True)
    data['SibSp'].replace(np.NaN,data['SibSp'].mean(),inplace=True)
    data['Parch'].replace(np.NaN,data['Parch'].mean(),inplace=True)
    data['Fare'].replace(np.NaN,data['Fare'].mean(),inplace=True)
    data['Embarked'].replace(['C','S','Q'],[1,2,3],inplace=True)
    data['Embarked'].replace(np.NaN,data['Embarked'].mean(),inplace=True)
    Datatest=data
    Xtest=np.array(Datatest[['Pclass','Sex','Age','Fare','SibSp','Parch','Embarked']]).T
    Xtest=normalise(Xtest)
    Pass=np.array(data['PassengerId']).reshape(1,Xtest.shape[1])
    return [Xtest,Pass]
def test(Xtest,W,B):
    Z=[Xtest]
    A=[Xtest]
    Z.append(np.dot(W[0],Xtest)+B[0])
    A.append(sigmoid(Z[-1]))
    Z.append(np.dot(W[1],A[-1])+B[1])
    A.append(sigmoid(Z[-1]))
    x=np.around(A[-1],0)
    return x
Xtrain,Ytrain,Xval,Yval=load_data()
L1,L2,W,B,alpha=initialize()
J=[]
print(trainonce(Xtrain,Ytrain,W,B,alpha))
for i in range(10000):
    J.append(trainonce(Xtrain,Ytrain,W,B,alpha)[0][0])
print(trainonce(Xtrain,Ytrain,W,B,alpha))
crossvalidate(Xval,Yval,W,B)
y=np.array(J)
plt.plot(y) 
plt.show()
Xtest,Pass=load_datatest()
predictions=test(Xtest,W,B)
print('PassengerId,Survived')
for i in range(predictions.shape[1]):
    print(Pass[0][i],int(predictions[0][i]),sep=",")
