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
rc_SpectralRadius 			= 0.11				# Spectral radius
rc_Size 					= 1000				# Taille du réservoir
rc_InputScaling 			= 0.51				# Dimensionnement des entrées
rc_LeakRate					= 0.41				# Leak rate
rc_Bias						= 1.0
rc_nbDigits					= 10				# Nombre de digits (sorties)
rc_TrainingLength			= 60000				# Longueur d'entrainement
rc_TestLength				= 10000				# Longeur de test
rc_ImagesSize				= 15				# Taille des digits
rc_SelectMethod 			= "average"			# Methode d'élection

####################################################
# Fonction principale
####################################################
if __name__ == "__main__":
	
	# Vérifie les params
	if len(sys.argv) != 2:
		print "Donnez un nom de fichier!"
		exit()

	# Import les digits
	print "Importation des digits depuis {}".format(sys.argv[1])
	digitImport = MNISTImporter()
	#digitImport = MNISTImporter.Open(sys.argv[1])
	
	digitImport.Load(sys.argv[1])
	digitImport.resizeImages(size = rc_ImagesSize)
	digitImport.addTimeserie()
	
	#digitImport.resetImages()
	digitImport.resizeImages(size = rc_ImagesSize)
	digitImport.rotateImages(angle = 30)
	digitImport.addTimeserie()
	
	#digitImport.resetImages()
	digitImport.resizeImages(size = rc_ImagesSize)
	digitImport.rotateImages(angle = 60)
	digitImport.addTimeserie()
	
	digitImport.generateLabels()
	digitImport.showTrainingSet()
	
	# Génère les labels
	outputs, outputs_test = digitImport.generateShortLabels()
	
	# Informations
	print "Longueur : " + str(digitImport.trainSetLength)
	print "Nb inputs : " + str(digitImport.nbInputs)
	
	# Réservoir et jointure
	reservoir = Oger.nodes.LeakyReservoirNode(input_dim = digitImport.nbInputs, output_dim = rc_Size, input_scaling = rc_InputScaling, spectral_radius = rc_SpectralRadius, leak_rate = rc_LeakRate, bias_scaling = rc_Bias)
	joiner = JoinedStatesNode(image_size = rc_ImagesSize, input_dim = rc_Size)
	
	# Regression et classifieur
	readout = Oger.nodes.RidgeRegressionNode(output_dim = rc_nbDigits, dtype='float64')
	classifier = DigitClassifierNode(mnist_space = 0, label_space_ratio = 0, digit_space_ratio = 0, image_size = 1, nb_digit = rc_nbDigits, method = "average", input_dim = rc_nbDigits, dtype='float64')
	
	# Récupère une partie du jeu d'entrainement et des labels
	inputs, out										= digitImport.getTrainingSet(length = rc_TrainingLength)
	inputs_test, out_test							= digitImport.getTestSet(length = rc_TestLength)
	
	# Informations
	print "Outputs : " + str(outputs.shape)
	print "Outputs : " + str(outputs.ndim)
	print "Inputs : " + str(inputs.shape)
	print "Inputs : " + str(inputs.ndim)

	# Calcule les états
	tmp_states = reservoir.execute(inputs)
	states = joiner.execute(tmp_states)
	digitImport.resetImages()
	digitImport.resizeImages(size = rc_ImagesSize)
	digitImport.trainImages.shape = (rc_TrainingLength, rc_ImagesSize*rc_ImagesSize)
	im_states = np.hstack((states, digitImport.trainImages))
	
	print "im_states : " + str(im_states.shape)
	
	# Flux de données
	data = [[(im_states, outputs)], None]
	
	# Construction du flux
	flow = mdp.Flow([readout, classifier], verbose=1)
	
	# Entrainement du réseau
	flow.train(data)
	
	# Calcule les états
	test_states = joiner.execute(reservoir.execute(inputs_test))
	digitImport.testImages.shape = (rc_TestLength, rc_ImagesSize*rc_ImagesSize)
	im_test_states = np.hstack((test_states, digitImport.testImages))
	print "im_test_states : " + str(im_test_states.shape)
	
	# Applique le réseau entraîné au jeu de test
	testout, out = flow(im_test_states)
	print "Testout : " + str(testout.shape)
	
	# Digit error rate
	der, misses, per_digit, miss_pos = digitImport.digitErrorRate(testout, with_miss_array = True, per_digit = True, pos_table = True)
	
	print "Digit Error Rate : {}".format(der)
	#print per_digit
	
