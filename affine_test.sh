#!/bin/bash

echo "Reference..."
time python simple_reservoir_joined_state.py ../../Datasets/MNIST/mnist.p load 60000
time python simple_reservoir_joined_state.py ../../Datasets/MNIST/mnist.p load 60000
time python simple_reservoir_joined_state.py ../../Datasets/MNIST/mnist.p load 60000
time python simple_reservoir_joined_state.py ../../Datasets/MNIST/mnist.p load 60000
time python simple_reservoir_joined_state.py ../../Datasets/MNIST/mnist.p load 60000
time python simple_reservoir_joined_state.py ../../Datasets/MNIST/mnist.p load 60000
time python simple_reservoir_joined_state.py ../../Datasets/MNIST/mnist.p load 60000
time python simple_reservoir_joined_state.py ../../Datasets/MNIST/mnist.p load 60000
time python simple_reservoir_joined_state.py ../../Datasets/MNIST/mnist.p load 60000
time python simple_reservoir_joined_state.py ../../Datasets/MNIST/mnist.p load 60000

echo "Testing first set..."
time python simple_reservoir_joined_state.py datasets/mnist_affine_1.p
time python simple_reservoir_joined_state.py datasets/mnist_affine_1.p
time python simple_reservoir_joined_state.py datasets/mnist_affine_1.p
time python simple_reservoir_joined_state.py datasets/mnist_affine_1.p
time python simple_reservoir_joined_state.py datasets/mnist_affine_1.p
time python simple_reservoir_joined_state.py datasets/mnist_affine_1.p
time python simple_reservoir_joined_state.py datasets/mnist_affine_1.p
time python simple_reservoir_joined_state.py datasets/mnist_affine_1.p
time python simple_reservoir_joined_state.py datasets/mnist_affine_1.p
time python simple_reservoir_joined_state.py datasets/mnist_affine_1.p

echo "Testing second set..."
time python simple_reservoir_joined_state.py datasets/mnist2_affine_1.p
time python simple_reservoir_joined_state.py datasets/mnist2_affine_1.p
time python simple_reservoir_joined_state.py datasets/mnist2_affine_1.p
time python simple_reservoir_joined_state.py datasets/mnist2_affine_1.p
time python simple_reservoir_joined_state.py datasets/mnist2_affine_1.p
time python simple_reservoir_joined_state.py datasets/mnist2_affine_1.p
time python simple_reservoir_joined_state.py datasets/mnist2_affine_1.p
time python simple_reservoir_joined_state.py datasets/mnist2_affine_1.p
time python simple_reservoir_joined_state.py datasets/mnist2_affine_1.p
time python simple_reservoir_joined_state.py datasets/mnist2_affine_1.p

echo "Testing third set..."
time python generate_training.py ../../Datasets/MNIST/mnist.p datasets/mnist3_affine_1.p 1
time python simple_reservoir_joined_state.py datasets/mnist3_affine_1.p
time python simple_reservoir_joined_state.py datasets/mnist3_affine_1.p
time python simple_reservoir_joined_state.py datasets/mnist3_affine_1.p
time python simple_reservoir_joined_state.py datasets/mnist3_affine_1.p
time python simple_reservoir_joined_state.py datasets/mnist3_affine_1.p
time python simple_reservoir_joined_state.py datasets/mnist3_affine_1.p
time python simple_reservoir_joined_state.py datasets/mnist3_affine_1.p
time python simple_reservoir_joined_state.py datasets/mnist3_affine_1.p
time python simple_reservoir_joined_state.py datasets/mnist3_affine_1.p
time python simple_reservoir_joined_state.py datasets/mnist3_affine_1.p
