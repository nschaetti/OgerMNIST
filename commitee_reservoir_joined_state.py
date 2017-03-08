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
import time

#########################################################################
#
# Ici on test un réservoir simple avec des neurones Leaky Integrator
#
#########################################################################

# Paramètres du réservoir
rc_SpectralRadius 			= 0.2				# Spectral radius
rc_Size 					= 1000				# Taille du réservoir
rc_InputScaling 			= 0.5				# Dimensionnement des entrées
rc_LeakRate					= 0.2				# Leak rate
rc_Bias						= 0.4
rc_nbDigits					= 10				# Nombre de digits (sorties)
rc_TrainingLength			= 60000				# Longueur d'entrainement
rc_TestLength				= 10000				# Longeur de test
rc_ImagesSize				= 15				# Taille des digits
rc_SelectMethod 			= "average"			# Methode d'élection
rc_nbMembers				= 50				# Nombre de membres dans le comité

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
	
	# Results
	results = np.array([])
	
	i = 0
	min_der = 100.0
	print "[C] Commitee error rate : {}".format(100)
	while i < rc_nbMembers:
		# Réservoir et jointure
		reservoir = Oger.nodes.LeakyReservoirNode(input_dim = digitImport.nbInputs, output_dim = rc_Size, input_scaling = rc_InputScaling, spectral_radius = rc_SpectralRadius, leak_rate = rc_LeakRate, bias_scaling = rc_Bias)
		joiner = JoinedStatesNode(image_size = rc_ImagesSize, input_dim = rc_Size)
		
		# Regression et classifieur
		readout = Oger.nodes.RidgeRegressionNode(output_dim = rc_nbDigits, dtype='float64')
		classifier = DigitClassifierNode(mnist_space = 0, label_space_ratio = 0, digit_space_ratio = 0, image_size = 1, nb_digit = rc_nbDigits, method = "average", input_dim = rc_nbDigits, dtype='float64')
		
		# Récupère une partie du jeu d'entrainement et des labels
		inputs, out										= digitImport.getTrainingSet(length = rc_TrainingLength)
		inputs_test, out_test							= digitImport.getTestSet(length = rc_TestLength)

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
		
		# Applique le réseau entraîné au jeu de test
		testout, out = flow(test_states)
		
		# Digit error rate
		der, misses, per_digit, miss_pos = digitImport.digitErrorRate(testout, with_miss_array = True, per_digit = True, pos_table = True)
		#print "Digit Error Rate for member {} : {}".format(i,der)
		
		# Calcul des votes
		if i == 0:
			results = testout.copy()
			results.shape = (rc_TestLength,1)
			i += 1
		else:
			testout.shape = (rc_TestLength,1)
			results = np.hstack((results,testout))
			
			# Calcul les votes
			votes = np.zeros(rc_TestLength, dtype='float64')
			for j in np.arange(0,rc_TestLength):
				votes[j] = np.argmax(np.bincount(np.array(results[j],dtype='int64')))
			der, misses, per_digit, miss_pos = digitImport.digitErrorRate(votes, with_miss_array = True, per_digit = True, pos_table = True)
			
			#print "[C] Commitee error rate : {}".format(der)
			
			if der < min_der:
				min_der = der
				print '\033[93m' + ">>>>>>>>> Digit Error Rate : {} for {} members <<<<<<<<<<<".format(der,i+1) + " (" + time.strftime('%d/%m/%y %H:%M',time.localtime()) + ") " + '\033[0m'
			
			i += 1
