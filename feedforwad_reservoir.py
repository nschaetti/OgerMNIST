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
rc_SpectralRadius 			= [0.2,0.2]					# Spectral radius
rc_Size 					= [100,1000]				# Taille du réservoir
rc_InputScaling 			= [0.2,0.2]					# Dimensionnement des entrées
rc_LeakRate					= [0.2,0.2]					# Leak rate
rc_Bias						= [0.4,0.4]
rc_nbDigits					= 10						# Nombre de digits (sorties)
rc_TrainingLength			= 60000						# Longueur d'entrainement
rc_TestLength				= 10000						# Longeur de test
rc_ImagesSize				= 22						# Taille des digits
rc_SelectMethod 			= "average"					# Methode d'élection

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
	digitImport.Load(sys.argv[1])
	
	# Série normale
	digitImport.resetImages()
	digitImport.resizeImages(size = rc_ImagesSize)
	digitImport.addTimeserie()
	
	# Série normale
	digitImport.rotateImages(angle = 30)
	digitImport.resizeImages(size = rc_ImagesSize)
	digitImport.addTimeserie()
	
	# Série normale
	digitImport.rotateImages(angle = 60)
	digitImport.resizeImages(size = rc_ImagesSize)
	digitImport.addTimeserie()
	
	# Ajoute les labels
	digitImport.generateLabels()
	
	# Génère les labels
	outputs, outputs_test = digitImport.generateShortLabels()
	
	# Informations
	print "Label généré..."
	print "Longueur du jeu d'entraînement : " + str(digitImport.trainSetLength)
	print "Nombre d'entrées : " + str(digitImport.nbInputs)
	print "Longueur d'une entrée : " + str(digitImport.entrySize)
	
	# Premier réservoir
	reservoir1 = Oger.nodes.LeakyReservoirNode(input_dim = digitImport.nbInputs, output_dim = rc_Size[0], input_scaling = rc_InputScaling[0], spectral_radius = rc_SpectralRadius[0], leak_rate = rc_LeakRate[0], bias_scaling = rc_Bias[0])
	
	# Deuxième réservoir
	reservoir2 = Oger.nodes.LeakyReservoirNode(input_dim = rc_Size[0], output_dim = rc_Size[1], input_scaling = rc_InputScaling[1], spectral_radius = rc_SpectralRadius[1], leak_rate = rc_LeakRate[1], bias_scaling = rc_Bias[1])
	joiner = JoinedStatesNode(image_size = digitImport.entrySize, input_dim = rc_Size[1])
	
	# Regression et classifieur
	readout = Oger.nodes.RidgeRegressionNode(output_dim = rc_nbDigits, dtype='float64')
	classifier = DigitClassifierNode(mnist_space = 0, label_space_ratio = 0, digit_space_ratio = 0, image_size = 1, nb_digit = rc_nbDigits, method = "average", input_dim = rc_nbDigits, dtype='float64')
	
	# Récupère une partie du jeu d'entrainement et des labels
	inputs, out					= digitImport.getTrainingSet(length = rc_TrainingLength)
	inputs_test, out_test		= digitImport.getTestSet(length = rc_TestLength)
	
	# Informations
	print "Génération des états du réservoir..."
	print "Outputs : " + str(outputs.shape)
	print "Outputs : " + str(outputs.ndim)
	print "Inputs : " + str(inputs.shape)
	print "Inputs : " + str(inputs.ndim)

	# Calcule les états du premier réservoir
	print "Génération du premier réservoir..."
	states = reservoir1.execute(inputs)
	print "Etats générés... forme : " + str(states.shape)
	
	# Calcule les états du deuxième réservoir
	print "Génération du deuxième réservoir..."
	states2 = joiner.execute(reservoir2.execute(states))
	print "Etats générés... forme : " + str(states2.shape)
	
	# Flux de données
	data = [[(states2, outputs)], None]
	
	# Construction du flux
	flow = mdp.Flow([readout, classifier], verbose=1)
	
	# Entrainement du réseau
	flow.train(data)
	
	# Calcule les états
	print "Génération des états sur le jeu de test..."
	test_states = joiner.execute(reservoir2.execute(reservoir1.execute(inputs_test)))
	
	# Applique le réseau entraîné au jeu de test
	testout, out = flow(test_states)
	print "Testout : " + str(testout.shape)
	
	# Digit error rate
	der, misses, per_digit, miss_pos = digitImport.digitErrorRate(testout, with_miss_array = True, per_digit = True, pos_table = True)
	
	print '\033[93m' + "Digit Error Rate : {}".format(der) + '\033[0m'
	
