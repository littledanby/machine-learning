import math
import random
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


fileRead = open("X.txt", "r")
X = fileRead.readlines()
fileRead.close()

fileRead = open("y.txt", "r")
y = fileRead.readlines()
fileRead.close()

for i in range(392):
	X[i] = X[i].split(",")
	for j in range(len(X[i])):
		X[i][j] = float(X[i][j])
	y[i] = float(y[i])


def prepare_data(x, y, p):
	xtest = []
	xtrain = []
	ytest = []
	ytrain = []
	index_list = range(0, 392)
	test_index = random.sample(index_list, 20)
	if p <= 1:
		a = x
	else:
		a = x[:]
		for i in range(2,p+1):
			for line in range(len(a)):
				temp = a[line][1:7]
				for k in range(len(temp)):
					temp[k] = temp[k]**p
				a[line].extend(temp)

	for i in range(392):
		if i in test_index:
			xtest.append(a[i])
			ytest.append(y[i])
		else:
			xtrain.append(a[i])
			ytrain.append(y[i])

	return xtrain, ytrain, xtest, ytest


def obtain_matrix(p):
	xtrain, ytrain, xtest, ytest = prepare_data(X, y, p)
	x_test = np.matrix(xtest)
	x_train = np.matrix(xtrain)
	y_test = np.matrix(ytest)
	y_train = np.matrix(ytrain)
	y_test = y_test.T
	y_train = y_train.T
	return x_train, y_train, x_test, y_test


def w_LS(x, y):
	m = 10**(-6) 
	return inv(x.T.dot(x) + np.eye(x.shape[1])*m).dot(x.T).dot(y)

def cal_y_pred(x, w):
	return x.dot(w)

def cal_MAE(x_train, y_train, x_test, y_test):
	w = w_LS(x_train, y_train)
	y_pred = cal_y_pred(x_test, w)
	return abs((y_test - y_pred)).sum() / len(y_test)

def cal_RMSE(x_train, y_train, x_test, y_test):
	w = w_LS(x_train, y_train)
	y_pred = cal_y_pred(x_test, w)
	return math.sqrt(np.square(y_test - y_pred).sum() / len(y_test))


def cal_RMSEset(p):
	RMSEset = []
	print "p is", p
	kk = 1
	for i in range(1000):
		print kk
		kk += 1
		x_train, y_train, x_test, y_test = obtain_matrix(p)
		RMSEset.append(cal_RMSE(x_train, y_train, x_test, y_test))
	#print "RMSEset is", RMSEset
	mean = np.mean(RMSEset)
	std = np.std(RMSEset)
	print "mean is", mean
	print "std is", std
	return mean, std


def get_error(p):
	error = []
	for i in range(99):
		x_train, y_train, x_test, y_test = obtain_matrix(p)
		w = w_LS(x_train, y_train)
		y_pred = cal_y_pred(x_test, w)
		a = list(np.array(y_test - y_pred).reshape(-1,))
		error.extend(a)
		
	return error



# part 1

def part1_a():
	x_train, y_train, x_test, y_test = obtain_matrix(0)
	print "w_LS is:", w_LS(x_train, y_train)


def part1_b():
	
	MAEset = []
	while len(MAEset) < 1000 :
		x_train, y_train, x_test, y_test = obtain_matrix(0)
		MAEset.append(cal_MAE(x_train, y_train, x_test, y_test))
	print "MAEset is", MAEset
	mean = np.mean(MAEset)
	std = np.std(MAEset)
	print "mean is", mean
	print "std is", std

#part 2

def part2_a():
	for p in range(1,5):
		mean, std = cal_RMSEset(p)
		print "p is",p
		print "mean is", mean
		print "std is", std


def part2_b(p):
	error = get_error(p)
	plt.hist(error, bins=50)
	plt.show()


def part2_c(p):
	error = get_error(p)
	N = len(error)
	mean = np.mean(error)
	var = np.var(error)
	t1 = -(N / 2) * math.log(2 * math.pi)
	t2 = -(N / 2) * math.log(var)
	t3 = -(1 / (2 * var)) * sum((x-mean)**2 for x in error)
	print "mean is", mean
	print "var is", var
	print "log likelihood is", t1+t2+t3

	
		





