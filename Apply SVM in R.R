---
title: "5241 HW3P3"
author: "MollyH"
date: "2018/3/20"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## In this problem, we will apply a support vector machine to classify hand-written digits. 
## You do not have to implement the SVM algorithm: The R library e1071 provides an implementation.

```{r}
setwd("/Users/minta/Desktop/5241 HW3 3.22")
d6=read.table("train.6.txt",header = F,sep=',')
d5=read.table("train.5.txt",header = F,sep=',')
x=rbind(as.matrix(d5),as.matrix(d6))
y=rep(c(-1,1),c(nrow(d5),nrow(d6)))

#Randomly select about 20% of the data and set it aside as a test set.
set.seed(1089)
h=nrow(x)*0.2
h
test.index<-sort(sample(1:nrow(x),h))
xtest=x[test.index,]
ytest=y[test.index]

xtrain=x[-test.index,]
ytrain=y[-test.index]
```
Linear Kernel
```{r}
set.seed(1089)
###Linear Kernel
#Train a linear SVM with soft margin. Cross-validate the margin parameter.
library(e1071)
traindata=data.frame(x=xtrain, y=as.factor(ytrain))
dim(traindata)
#By default, tune() performs ten-fold cross-validation
tune.out=tune(svm, y~.,data=traindata,kernel="linear",
              ranges=list(cost=10^(seq(-5, 2, 0.5))),scale=FALSE)
#cross-validation error rates
summary(tune.out)
#plot the misclassification rates as a function of the margin parameter in the linear case
plot(tune.out,main="Linear Kernel")

#best model
bestmod=tune.out$best.model 
summary(bestmod)

#predict the class label on a set of test observations
testdata=data.frame(x=xtest, y=as.factor(ytest))
#train with the Cost parameter=0.03162278 selected via CV
svmfit=svm(y~., data=traindata, kernel="linear", cost=0.03162278, scale=FALSE) 
ypred1=predict(svmfit,testdata) #method1
ypred2=predict(bestmod,testdata) #method2
#test error
sum(ypred1 != ytest)/length(ytest)
sum(ypred2 != ytest)/length(ytest) #1.64% of test observations are misclassified by this SVM.
```

RBF Kernel
```{r}
set.seed(1089)
###RBF Kernel
#Train an SVM with soft margin and RBF kernel.
#cross-validate both the soft-margin parameter and the kernel bandwidth.
traindata=data.frame(x=xtrain, y=as.factor(ytrain))
tune.out1=tune(svm, y~., data=traindata, kernel="radial",
ranges=list(cost=10^(seq(-5, 2, 0.5)),gamma=c(0.001,0.01,0.1,1)),scale=FALSE)
#cross-validation error rates
summary(tune.out1)
#plot the misclassification rates a function of the margin parameter and the kernel bandwidth
plot(tune.out1,main="RBF Kernel")

#best model
bestmod1=tune.out1$best.model 
summary(bestmod1)

#predict the class label on a set of test observations
testdata=data.frame(x=xtest, y=as.factor(ytest))
#train with the parameters(cost=1  gamma=0.01) selected via CV
svmfit1=svm(y~., data=traindata, kernel="radial",gamma=0.01, cost=1, scale=FALSE) 
ypred11=predict(svmfit1,testdata) #method1
ypred22=predict(bestmod1,testdata) #method2
#test error
sum(ypred11 != ytest)/length(ytest)
sum(ypred22 != ytest)/length(ytest) #0.41% of test observations are misclassified by this SVM.
```
Since 1.64% of test observations are misclassified by using Linear SVM, and 0.41% of test observations are misclassified by using non-linear SVM, we should use the non-linear one.








