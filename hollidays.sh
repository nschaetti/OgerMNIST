#!/bin/bash

## Etude de la longueur du jeu d'entrainement
#echo "##### Longueur du jeu d'entrainement :"
#for len in {1000..60000..1000}
#do
	#echo "Longueur du set $len"
	#for i in {1..10}
	#do
		#time python simple_reservoir_joined_state.py ./datasets/mnist_hollidays.p open $len 2> /dev/null
	#done
	#echo "Fin"
#done

## Réseau small-world
#echo "##### Small-world network"
#for degree in {5..75..10}
#do
	#for beta in {0..10..2}
	#do
		#echo "degree = $degree, beta = $beta"
		#for i in {0..10}
		#do
			#python small_world_search.py ./datasets/mnist_hollidays.p open 60000 $degree $beta 2> /dev/null
		#done
	#done
#done

## Réseau scale-free
#echo "##### Scale-free network"
#for m in {1..91..10}
#do
	#echo "m = $m"
	#for i in {0..10}
	#do
		#python scale_free_search.py ./datasets/mnist_hollidays.p open 60000 $m 2> /dev/null
	#done
#done

# Transformations affines
echo "##### Transformations affines..."
for scale_dev in {0..3..1}
do
	for rotation_dev in {0..12..4}
	do
		for shear_dev in {0..9..3}
		do
			echo "scale_dev = $scale_dev, rotation_dev = $rotation_dev, shear_dev = $shear_dev"
			python generate_training.py ../../Datasets/MNIST/mnist.p ./datasets/mnist_tmp.p 1 affine $scale_dev $rotation_dev $shear_dev > /dev/null
			for i in {1..10}
			do
				python simple_reservoir_joined_state.py ./datasets/mnist_tmp.p open 120000 100 
			done
			rm ./datasets/mnist_tmp.p
		done
	done
done

# Transformation elastiques
echo "##### Transformations elastiques..."
for scale in {1..30..10}
do
	for sigma in {0..9..3}
	do
		echo "scale = $scale, sigma = $sigma"
		python generate_training.py ../../Datasets/MNIST/mnist.p ./datasets/mnist_tmp.p 1 elastic $scale $sigma > /dev/null
		for i in {1..10}
		do
			python simple_reservoir_joined_state.py ./datasets/mnist_tmp.p open 120000 100
		done
		rm ./datasets/mnist_tmp.p
	done
done

# Training length pour 1000N
echo "##### Training length pour 1000N"
for len in {5000..60000..5000}
do
	echo "training length = $len"
	time python simple_reservoir_joined_state.py ./datasets/mnist_hollidays.p open $len 1000 2> /dev/null 
done

# 2kN, 3kN, 4kN, 5kN, avec train_length = 35000
echo "##### 2kN, 3kN, 4kN, 5kN, avec train_length = 35000"
for size in {2000..5000..1000}
do
	echo "size = $size"
	time python simple_reservoir_joined_state.py ./datasets/mnist_hollidays.p open 30000 $size 2> /dev/null 
done
