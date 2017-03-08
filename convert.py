import cPickle
import struct
import sys

from numpy import *

path = sys.argv[1]

trainfile = open(path + 'train-images.idx3-ubyte', 'r')
testfile = open(path + 't10k-images.idx3-ubyte', 'r')
trainlabfile = open(path + 'train-labels.idx1-ubyte', 'r')
testlabfile = open(path + 't10k-labels.idx1-ubyte', 'r')

trainfile.read(16)
trainimages = zeros((60000, 784))
line = trainfile.read(28 * 28 * 60000)
for i in range(60000):
	for j in range(784):
		trainimages[i, j] = int(struct.unpack('B', line[i * 784 + j])[0])
	trainimages[i, :]

testfile.read(16)
testimages = zeros((10000, 784))
line = testfile.read(28 * 28 * 10000)
for i in range(10000):
	for j in range(784):
		testimages[i, j] = int(struct.unpack('B', line[i * 784 + j])[0])
	testimages[i, :]

trainlabfile.read(8)
trainlabels = zeros(60000)
line = trainlabfile.read(60000)
for i in range(60000):
	trainlabels[i] = int(struct.unpack('B', line[i])[0])

testlabfile.read(8)
testlabels = zeros(10000)
line = testlabfile.read(10000)
for i in range(10000):
	testlabels[i] = int(struct.unpack('B', line[i])[0])

data = dict()

data['testlabels'] = testlabels
data['trainlabels'] = trainlabels
data['testimages'] = testimages
data['trainimages'] = trainimages

cPickle.dump(data, open('mnist.p', 'w'), protocol=2)
