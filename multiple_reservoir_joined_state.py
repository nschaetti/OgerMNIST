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
rc_Size 					= 200				# Taille du réservoir
rc_InputScaling 			= 0.2				# Dimensionnement des entrées
rc_LeakRate					= 0.2				# Leak rate
rc_Bias						= 0.4
rc_nbDigits					= 10				# Nombre de digits (sorties)
rc_TrainingLength			= 60000				# Longueur d'entrainement
rc_TestLength				= 10000				# Longeur de test
rc_ImagesSize				= 22				# Taille des digits
rc_SelectMethod 			= "average"			# Methode d'élection
rc_NbReservoirs				= 2

####################################################
# Fonction principale
####################################################
if __name__ == "__main__":
	
	# Vérifie les params
	if len(sys.argv) != 2:
		print "Donnez un nom de fichier!"
		exit()
	
	# Regression et classifieur
	readout = Oger.nodes.RidgeRegressionNode(output_dim = rc_nbDigits, dtype='float64')
	classifier = DigitClassifierNode(mnist_space = 0, label_space_ratio = 0, digit_space_ratio = 0, image_size = 1, nb_digit = rc_nbDigits, method = "average", input_dim = rc_nbDigits, dtype='float64')
	
	# Pour chaque réservoirs
	for i in range(rc_NbReservoirs):
		
		# Infos
		print "Réservoirs {}".format(i)
		
		# Import les digits
		print "Importation des digits depuis {}".format(sys.argv[1])
		digitImport = MNISTImporter()	
		digitImport.Load(sys.argv[1])
		
		# Entrées par réservoirs
		if i == 0:
			# Série normale
			digitImport.resetImages()
			digitImport.resizeImages(size = rc_ImagesSize)
			digitImport.addTimeserie()
		elif i == 1:
			# Série normale
			digitImport.rotateImages(angle = 30)
			digitImport.resizeImages(size = rc_ImagesSize)
			digitImport.addTimeserie()
		elif i == 2:
			# Série normale
			digitImport.rotateImages(angle = 60)
			digitImport.resizeImages(size = rc_ImagesSize)
			digitImport.addTimeserie()
		
		# Ajoute les labels
		digitImport.generateLabels()
		
		# Récupère une partie du jeu d'entrainement et des labels
		inputs, out					= digitImport.getTrainingSet(length = rc_TrainingLength)
		
		# Réservoir et jointure
		reservoir = Oger.nodes.LeakyReservoirNode(input_dim = digitImport.nbInputs, output_dim = round(rc_Size/rc_NbReservoirs), input_scaling = rc_InputScaling, spectral_radius = rc_SpectralRadius, leak_rate = rc_LeakRate, bias_scaling = rc_Bias)
		joiner = JoinedStatesNode(image_size = digitImport.entrySize, input_dim = round(rc_Size/rc_NbReservoirs))

		# Calcule les états
		tmp_states = reservoir.execute(inputs)
		states = joiner.execute(tmp_states)
		
		# Jointure
		if i == 0:
			total_states = states
		else:
			total_states = np.hstack((total_states,states))
		
		# Informations
		print "Etats générés... forme : " + str(total_states.shape)
	
	# Génère les labels
	outputs, outputs_test = digitImport.generateShortLabels()
	
	# Flux de données
	data = [[(total_states, outputs)], None]
	
	# Construction du flux
	flow = mdp.Flow([readout, classifier], verbose=1)
	
	# Entrainement du réseau
	flow.train(data)
	
	# Pour chaque réservoirs
	for i in range(rc_NbReservoirs):
		
		# Infos
		print "Réservoirs {}".format(i)
		
		# Import les digits
		print "Importation des digits depuis {}".format(sys.argv[1])
		digitImport = MNISTImporter()	
		digitImport.Load(sys.argv[1])
		
		# Entrées par réservoirs
		if i == 0:
			# Série normale
			digitImport.resetImages()
			digitImport.resizeImages(size = rc_ImagesSize)
			digitImport.addTimeserie()
		elif i == 1:
			# Série normale
			digitImport.rotateImages(angle = 30)
			digitImport.resizeImages(size = rc_ImagesSize)
			digitImport.addTimeserie()
		elif i == 2:
			# Série normale
			digitImport.rotateImages(angle = 60)
			digitImport.resizeImages(size = rc_ImagesSize)
			digitImport.addTimeserie()
		
		# Ajoute les labels
		digitImport.generateLabels()
		
		# Récupère une partie du jeu d'entrainement et des labels
		inputs_test, out_test		= digitImport.getTestSet(length = rc_TestLength)
		
		# Calcule les états
		test_states = joiner.execute(reservoir.execute(inputs_test))
		
		# Jointure
		if i == 0:
			total_states_test = test_states
		else:
			total_states_test = np.hstack((total_states_test,test_states))
		
		# Informations
		print "Etats générés... forme : " + str(total_states_test.shape)
	
	# Applique le réseau entraîné au jeu de test
	testout, out = flow(total_states_test)
	print "Testout : " + str(testout.shape)
	
	# Digit error rate
	der, misses, per_digit, miss_pos = digitImport.digitErrorRate(testout, with_miss_array = True, per_digit = True, pos_table = True)
	
	print '\033[93m' + "Digit Error Rate : {}".format(der) + '\033[0m'
	
