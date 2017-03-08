#!/bin/bash

# Generate more example
echo "Generating more examples..."
time python generate_training.py ../../Datasets/MNIST/mnist.p datasets/mnist3_elastic_1.p 1 elastic 36 6
time python generate_training.py ../../Datasets/MNIST/mnist.p datasets/mnist4_elastic_1.p 1 elastic 18 3

echo "Testing first set..."
time python simple_reservoir_joined_state.py datasets/mnist3_elastic_1.p
time python simple_reservoir_joined_state.py datasets/mnist3_elastic_1.p
time python simple_reservoir_joined_state.py datasets/mnist3_elastic_1.p
time python simple_reservoir_joined_state.py datasets/mnist3_elastic_1.p
time python simple_reservoir_joined_state.py datasets/mnist3_elastic_1.p
time python simple_reservoir_joined_state.py datasets/mnist3_elastic_1.p
time python simple_reservoir_joined_state.py datasets/mnist3_elastic_1.p
time python simple_reservoir_joined_state.py datasets/mnist3_elastic_1.p
time python simple_reservoir_joined_state.py datasets/mnist3_elastic_1.p
time python simple_reservoir_joined_state.py datasets/mnist3_elastic_1.p

echo "Testing second set..."
time python simple_reservoir_joined_state.py datasets/mnist4_elastic_1.p
time python simple_reservoir_joined_state.py datasets/mnist4_elastic_1.p
time python simple_reservoir_joined_state.py datasets/mnist4_elastic_1.p
time python simple_reservoir_joined_state.py datasets/mnist4_elastic_1.p
time python simple_reservoir_joined_state.py datasets/mnist4_elastic_1.p
time python simple_reservoir_joined_state.py datasets/mnist4_elastic_1.p
time python simple_reservoir_joined_state.py datasets/mnist4_elastic_1.p
time python simple_reservoir_joined_state.py datasets/mnist4_elastic_1.p
time python simple_reservoir_joined_state.py datasets/mnist4_elastic_1.p
time python simple_reservoir_joined_state.py datasets/mnist4_elastic_1.p
