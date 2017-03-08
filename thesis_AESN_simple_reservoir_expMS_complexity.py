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
rc_SpectralRadius 			= 0.4				# Spectral radius
rc_InputScaling 			= 0.6																# Dimensionnement des entrées
rc_LeakRate					= 0.3							# Leak rate
rc_InputSparsity			= 0.9
rc_ReservoirSparsity		= 0.1
rc_Biais					= 1.0

rc_Size 					= [100,200,300,400,500,600,700,800,900,1000,1100,1200]				# Taille du réservoir
rc_nbDigits					= 10				# Nombre de digits (sorties)
rc_TrainingLength			= 60000				# Longueur d'entrainement
rc_TestLength				= 10000				# Longeur de test
rc_ImagesSize				= 15				# Taille des digits
rc_SelectMethod 			= "average"			# Methode d'élection
rc_SampleSize				= 1

####################################################
# Fonction principale
####################################################
if __name__ == "__main__":
	
	# Vérifie les params
	if len(sys.argv) < 2:
		print "Donnez un nom de fichier!"
		exit()

	# Import les digits
	print "Importation des digits depuis {}".format(sys.argv[1])
	digitImport = MNISTImporter()
	digitImport.Load(sys.argv[1])
	
	# Ajouteu une série temporelle
	digitImport.resizeImages(size = rc_ImagesSize)
	digitImport.addTimeserie()
	
	digitImport.rotateImages(angle = 30)
	digitImport.resizeImages(size = rc_ImagesSize)
	digitImport.addTimeserie()
	
	digitImport.rotateImages(angle = 60)
	digitImport.resizeImages(size = rc_ImagesSize)
	digitImport.addTimeserie()
	
	# Génère les labels
	digitImport.generateLabels()
	outputs, outputs_test = digitImport.generateShortLabels(length = rc_TrainingLength)
	
	for size in rc_Size:
		
		print "Start at " + time.strftime('%d/%m/%y %H:%M:%S',time.localtime())
		
		# Réservoir et jointure
		reservoir = Oger.nodes.LeakyReservoirNode(input_dim = digitImport.nbInputs, output_dim = size, input_scaling = rc_InputScaling, spectral_radius = rc_SpectralRadius, leak_rate = rc_LeakRate, bias_scaling = rc_Biais)
		joiner = MixedStateNode(image_size = digitImport.entrySize, input_dim = size)
		
		# Regression et classifieur
		readout = Oger.nodes.RidgeRegressionNode(output_dim = rc_nbDigits, dtype='float64')
		classifier = DigitClassifierNode(mnist_space = 0, label_space_ratio = 0, digit_space_ratio = 0, image_size = 1, nb_digit = rc_nbDigits, method = "average", input_dim = rc_nbDigits, dtype='float64')
		
		# Récupère une partie du jeu d'entrainement et des labels
		inputs, out					= digitImport.getTrainingSet(length = rc_TrainingLength)
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
		#print "Testout : " + str(testout.shape)
		
		# Digit error rate
		der, misses, per_digit, miss_pos = digitImport.digitErrorRate(testout, with_miss_array = True, per_digit = True, pos_table = True)
		
		print '\033[93m' + "Finish at {} with paramètre rc_SpectralRadius={},rc_LeakRate={},size={} : moyenne de {}".format(time.strftime('%d/%m/%y %H:%M:%S',time.localtime()),rc_SpectralRadius, rc_LeakRate, size, der) + '\033[0m'
	
	
