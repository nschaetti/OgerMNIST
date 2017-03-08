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
rc_SpectralRadius 			= 0.01				# Spectral radius
rc_Size 					= 3000				# Taille du réservoir
rc_InputScaling 			= 0.5				# Dimensionnement des entrées
rc_LeakRate					= 0.4				# Leak rate
rc_nbDigits					= 10				# Nombre de digits (sorties)
rc_TrainingLength			= 60000				# Longueur d'entrainement
rc_TestLength				= 10000				# Longeur de test
rc_ImagesSize				= 22				# Taille des digits
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
	digitImport.Load(sys.argv[1])
	digitImport.resizeImages(rc_ImagesSize)
	digitImport.addTimeserie()
	digitImport.rotateImages(angle = 10)
	digitImport.addTimeserie()
	digitImport.rotateImages(angle = 10)
	digitImport.addTimeserie()
	digitImport.rotateImages(angle = 10)
	digitImport.addTimeserie()
	digitImport.rotateImages(angle = 10)
	digitImport.addTimeserie()
	digitImport.rotateImages(angle = 10)
	digitImport.addTimeserie()
	digitImport.rotateImages(angle = 10)
	digitImport.addTimeserie()
	digitImport.rotateImages(angle = 10)
	digitImport.addTimeserie()
	digitImport.rotateImages(angle = 10)
	digitImport.addTimeserie()
	digitImport.rotateImages(angle = 10)
	digitImport.addTimeserie()
	digitImport.resetImages()
	digitImport.resizeImages(rc_ImagesSize)
	digitImport.integralImages()
	digitImport.addTimeserie()
	digitImport.generateLabels()
	digitImport.showTrainingSet()
	
	# Créer du réservoir
	reservoir = Oger.nodes.LeakyReservoirNode(input_dim = digitImport.nbInputs, output_dim = rc_Size, input_scaling = rc_InputScaling, spectral_radius = rc_SpectralRadius, leak_rate = rc_LeakRate)
	readout = Oger.nodes.RidgeRegressionNode(output_dim = rc_nbDigits, dtype='float64')
	classifier = DigitClassifierNode(mnist_space = digitImport.interImagesSpace, label_space_ratio = digitImport.interImagesRatio, digit_space_ratio = digitImport.digitImageRatio, image_size = digitImport.imagesSize, nb_digit = rc_nbDigits, method = "average", input_dim = rc_nbDigits, dtype='float64')
	
	# Récupère une partie du jeu d'entrainement et des labels
	inputs, outputs						= digitImport.getTrainingSet(length = rc_TrainingLength)
	inputs_test, outputs_test			= digitImport.getTestSet(length = rc_TestLength)
	data = [None, [(inputs, outputs)], None]
	
	# Construction du flux
	flow = mdp.Flow([reservoir, readout, classifier], verbose=0)
	
	# Entrainement du réseau
	flow.train(data)
	
	# Applique le réseau entraîné au jeu de test
	testout, out = flow(inputs_test)

	# Digit error rate
	der = float(digitImport.digitErrorRate(testout))
	
	print "Digit Error Rate : {}".format(der)
