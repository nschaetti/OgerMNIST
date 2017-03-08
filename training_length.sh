#!/bin/bash

for len in {1000..60000..1000}
do
	echo "Longueur du set $len"
	for i in {1..10}
	do
		time python simple_reservoir_joined_state.py ../../Datasets/MNIST/mnist.p load $len 2> /dev/null
	done
	echo "Fin"
done
