#!/usr/bin/env python
# -*- coding: utf-8 -*-

##########################################################################
#
# Auteur : Nils Schaetti <nils.schaetti@univ-comte.fr>
# Date : 19.04.2015 17:59:05
# Lieu : Nyon, Suisse
# 
# Fichier sous licence GNU GPL
#
###########################################################################

import Oger
import pylab
import mdp
import os
import cPickle
import struct
import sys
from mnist import *
from nodes import *
from DigitReservoir import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm

#########################################################################
#
# Ici on test un réservoir simple avec des neurones Leaky Integrator
#
#########################################################################

# Paramètres du réservoir
rc_SpectralRadius 			= 0.2				# Spectral radius
rc_Size 					= [100,200,300,400,500,600,700,800,900,1000,1100,1200]				# Taille du réservoir
rc_InputScaling 			= 0.5				# Dimensionnement des entrées
rc_LeakRate					= 0.8				# Leak rate
rc_Bias						= 0.4

rc_nbDigits					= 10				# Nombre de digits (sorties)
rc_TrainingLength			= 60000				# Longueur d'entrainement
rc_TestLength				= 10000				# Longeur de test
rc_ImagesSize				= 15				# Taille des digits
rc_SelectMethod 			= "average"			# Methode d'élection
rc_Start					= 5
rc_NbSteps					= 10

rc_SampleSize				= 100

####################################################
# Fonction principale
####################################################
if __name__ == "__main__":
	
	# Vérifie les params
	if len(sys.argv) < 2:
		print "Donnez un nom de fichier!"
		exit()

	# Import les digits
	digitImport = MNISTImporter()
	digitImport.Load(sys.argv[1])
	
	# Resize
	digitImport.resizeImages(size = rc_ImagesSize)
	
	# Série ondelettes
	digitImport.addTimeserieOndelette(start = rc_Start, end = rc_ImagesSize, nbsteps = rc_NbSteps)
	
	# Ajoute les labels
	digitImport.generateImageLabels()
	
	for training_length in rc_TrainingLength:
		
		# Génère les labels
		outputs, outputs_test = digitImport.generateShortLabels(length = training_length)
		
		# Réservoir et jointure
		reservoir = Oger.nodes.LeakyReservoirNode(input_dim = digitImport.nbInputs, output_dim = size, input_scaling = rc_InputScaling, spectral_radius = rc_SpectralRadius, leak_rate = rc_LeakRate, bias_scaling = rc_Bias)
		joiner = MixedThreeStateNode(image_size = rc_NbSteps, input_dim = size)
		
		# Readouts
		readout = Oger.nodes.RidgeRegressionNode(output_dim = rc_nbDigits, dtype = 'float64')
		classifier = DigitClassifierNode(mnist_space = 0, label_space_ratio = 0, digit_space_ratio = 0, image_size = 1, nb_digit = rc_nbDigits, method = 'average', input_dim = rc_nbDigits, dtype = 'float64')
		
		# Récupère une partie du jeu d'entrainement et des labels
		inputs, out					= digitImport.getTrainingSet(length = training_length)
		inputs_test, out_test		= digitImport.getTestSet(length = rc_TestLength)
		
		# Calcule les états
		tmp_states = reservoir.execute(inputs)
		states = joiner.execute(tmp_states)
		
		# Flux de données
		data = [[(states, outputs)], None]
		
		# Construction du flux
		flow = mdp.Flow([readout, classifier], verbose=0)
		
		# Entrainement du réseau
		flow.train(data)
		
		# Calcule les états
		test_states = joiner.execute(reservoir.execute(inputs_test))
		
		# Applique le réseau entraîné au jeu de test et au jeu d'entrainement
		testout, out = flow(test_states)
		trainout, _ = flow(states)
		
		# Digit error rate
		der, misses, per_digit, miss_pos = digitImport.digitErrorRate(testout, with_miss_array = True, per_digit = True, pos_table = True)
		print '\033[93m' + "rc_SpectralRadius={},rc_LeakRate={},rc_TrainingLength={} ::: {}".format(rc_SpectralRadius,rc_LeakRate,rc_TrainingLength,der) + '\033[0m'
		
	
