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
rc_Size 					= np.arange(100,1201,100)				# Taille du réservoir
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

rc_SampleSize				= 20

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
	
	print "Données créées..."
	
	for size in rc_Size:
		# Réservoir et jointure
		reservoir = Oger.nodes.LeakyReservoirNode(input_dim = digitImport.nbInputs, output_dim = size, input_scaling = rc_InputScaling, spectral_radius = rc_SpectralRadius, leak_rate = rc_LeakRate, bias_scaling = rc_Bias)
		readout = Oger.nodes.RidgeRegressionNode(output_dim = rc_nbDigits, dtype = 'float64')
		classifier = DigitClassifierNode(mnist_space = digitImport.interImagesSpace, label_space_ratio = digitImport.interImagesRatio, digit_space_ratio = digitImport.digitImageRatio, image_size = rc_NbSteps, nb_digit = rc_nbDigits, method = 'average', input_dim = rc_nbDigits, dtype = 'float64')
		
		# Récupère une partie du jeu d'entrainement et des labels
		inputs, out					= digitImport.getTrainingSet(length = rc_TrainingLength)
		inputs_test, out_test		= digitImport.getTestSet(length = rc_TestLength)
		
		# Données
		data = [None, [(inputs, out)], None]
		
		# Construction du flux
		flow = mdp.Flow([reservoir, readout, classifier], verbose=0)
		
		# Entrainement du réseau
		flow.train(data)
		
		# Applique le réseau entraîné au jeu de test et au jeu d'entrainement
		testout, out = flow(inputs_test)
		
		# Digit error rate
		der, misses, per_digit, miss_pos = digitImport.digitErrorRate(testout, with_miss_array = True, per_digit = True, pos_table = True)
		print '\033[93m' + "rc_SpectralRadius={},rc_LeakRate={},rc_Size={} : moyenne de {}".format(rc_SpectralRadius,rc_LeakRate,size,der) + '\033[0m'
		
	
