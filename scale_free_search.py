#!/usr/bin/env python
# -*- coding: utf-8 -*-

##########################################################################
#
# Auteur : Nils Schaetti <nils.schaetti@univ-comte.fr>
# Date : 17.06.2015 19:32:14
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
from topology import *

#########################################################################
#
# Ici on test un réservoir simple avec des neurones Leaky Integrator
#
#########################################################################

# Paramètres du réservoir
rc_SpectralRadius 			= 0.2				# Spectral radius
rc_Size 					= 100				# Taille du réservoir
rc_InputScaling 			= 0.5				# Dimensionnement des entrées
rc_LeakRate					= 0.2				# Leak rate
rc_Bias						= 0.4
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
	if len(sys.argv) < 2:
		print "Donnez un nom de fichier!"
		exit()

	# Import les digits
	if len(sys.argv) >= 3 and sys.argv[2] == "load":
		#print "Importation des digits depuis {}".format(sys.argv[1])
		digitImport = MNISTImporter()
		digitImport.Load(sys.argv[1])
	else:
		digitImport = MNISTImporter.Open(sys.argv[1])
	if len(sys.argv) >= 4:
		rc_TrainingLength = int(sys.argv[3])
		
	m = int(sys.argv[4])
	
	# Série normale
	#digitImport.resetImages()
	#digitImport.resizeImages(size = rc_ImagesSize)
	#digitImport.addTimeserie()
	
	## Série normale
	#digitImport.rotateImages(angle = 30)
	#digitImport.resizeImages(size = rc_ImagesSize)
	#digitImport.addTimeserie()
	
	## Série normale
	#digitImport.rotateImages(angle = 60)
	#digitImport.resizeImages(size = rc_ImagesSize)
	#digitImport.addTimeserie()
	
	## Ajoute les labels
	#digitImport.generateLabels()
	
	# Génère les labels
	outputs, outputs_test = digitImport.generateShortLabels(length = rc_TrainingLength)
	
	# Informations
	#print "Label généré..."
	#print "Longueur du jeu d'entraînement : " + str(digitImport.trainSetLength)
	#print "Nombre d'entrées : " + str(digitImport.nbInputs)
	#print "Longueur d'une entrée : " + str(digitImport.entrySize)
	
	# Crée le réseau small-world
	w = createScaleFreeNetwork(size = rc_Size, m0 = m, m = m)
	##w *= rc_SpectralRadius / Oger.utils.get_spectral_radius(w)
	
	# Réservoir et jointure
	##reservoir = Oger.nodes.LeakyReservoirNode(w = w, input_dim = digitImport.nbInputs, output_dim = rc_Size, input_scaling = rc_InputScaling, spectral_radius = rc_SpectralRadius, leak_rate = rc_LeakRate, bias_scaling = rc_Bias)
	##joiner = JoinedStatesNode(image_size = digitImport.entrySize, input_dim = rc_Size)
	
	# Regression et classifieur
	##readout = Oger.nodes.RidgeRegressionNode(output_dim = rc_nbDigits, dtype='float64')
	##classifier = DigitClassifierNode(mnist_space = 0, label_space_ratio = 0, digit_space_ratio = 0, image_size = 1, nb_digit = rc_nbDigits, method = "average", input_dim = rc_nbDigits, dtype='float64')
	
	# Récupère une partie du jeu d'entrainement et des labels
	##inputs, out					= digitImport.getTrainingSet(length = rc_TrainingLength)
	##inputs_test, out_test		= digitImport.getTestSet(length = rc_TestLength)
	
	# Informations
	#print "Génération des états du réservoir..."
	#print "Outputs : " + str(outputs.shape)
	#print "Outputs : " + str(outputs.ndim)
	#print "Inputs : " + str(inputs.shape)
	#print "Inputs : " + str(inputs.ndim)

	# Calcule les états
	##tmp_states = reservoir.execute(inputs)
	##states = joiner.execute(tmp_states)
	
	# Informations
	#print "Etats générés... forme : " + str(states.shape)
	
	# Flux de données
	##data = [[(states, outputs)], None]
	
	# Construction du flux
	##flow = mdp.Flow([readout, classifier], verbose=0)
	
	# Entrainement du réseau
	##flow.train(data)
	
	# Calcule les états
	##test_states = joiner.execute(reservoir.execute(inputs_test))
	
	# Applique le réseau entraîné au jeu de test et au jeu d'entrainement
	##testout, out = flow(test_states)
	##trainout, _ = flow(states)
	#print "Testout : " + str(testout.shape)
	
	# Digit error rate
	##der, misses, per_digit, miss_pos = digitImport.digitErrorRate(testout, with_miss_array = True, per_digit = True, pos_table = True)
	##print '\033[93m' + "Digit Error Rate : {}".format(der) + '\033[0m'
	
	# Digit error rate
	##der, misses, per_digit, miss_pos = digitImport.digitErrorRateTrainging(trainout, with_miss_array = True, per_digit = True, pos_table = True)
	##print '\033[93m' + "Training Error Rate : {}".format(der) + '\033[0m'
	
	
