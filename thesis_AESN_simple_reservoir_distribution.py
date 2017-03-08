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
# Ici on test un réservoir simple
#
#########################################################################

# Paramètres du réservoir
rc_SpectralRadius 			= 1.3				# Spectral radius
rc_InputScaling 			= 0.6				# Dimensionnement des entrées
rc_LeakRate					= 0					# Leak rate
rc_InputSparsity			= 0.9
rc_ReservoirSparsity		= 0.1
rc_Biais					= 1.0

rc_Size 					= 100				# Taille du réservoir
rc_nbDigits					= 10				# Nombre de digits (sorties)
rc_TrainingLength			= 60000				# Longueur d'entrainement
rc_TestLength				= 10000				# Longeur de test
rc_ImagesSize				= 28				# Taille des digits
rc_SelectMethod 			= "average"			# Methode d'élection
rc_SampleSize				= 1000

####################################################
# Fonction principale
####################################################
if __name__ == "__main__":
	
	# Vérifie les params
	if len(sys.argv) != 2:
		print "Donnez un nom de fichier!"
		exit()

	# Import les digits
	#print "Importation des digits depuis {}".format(sys.argv[1])
	digitImport = MNISTImporter()
	digitImport.Load(sys.argv[1])
	
	# Ajouteu une série temporelle
	digitImport.addTimeserie()
	
	# Génère les labels
	digitImport.generateLabels()
	
	# Construction des noeuds
	results = np.zeros(rc_SampleSize)
	for n in np.arange(0,rc_SampleSize):
		#print "Création d'un réservoir de taille {}, avec {} sortie(s) et {} entrée(s)!".format(rc_Size, rc_nbDigits, rc_ImagesSize, dtype = 'float64')
		reservoir = Oger.nodes.ReservoirNode(input_dim = rc_ImagesSize, output_dim = rc_Size, input_scaling = rc_InputScaling, spectral_radius = rc_SpectralRadius, bias_scaling=rc_Biais, sparsity = rc_InputSparsity, w_sparsity = rc_ReservoirSparsity)
		readout = Oger.nodes.RidgeRegressionNode(output_dim = rc_nbDigits, dtype = 'float64')
		classifier = DigitClassifierNode(mnist_space = digitImport.interImagesSpace, label_space_ratio = digitImport.interImagesRatio, digit_space_ratio = digitImport.digitImageRatio, image_size = rc_ImagesSize, nb_digit = rc_nbDigits, method = 'average', input_dim = rc_nbDigits, dtype = 'float64')
		
		# Construction du flux
		#print "Construction du flux..."
		flow = mdp.Flow([reservoir, readout, classifier], verbose=0)
		#Oger.utils.make_inspectable(Oger.nodes.ReservoirNode)
		
		# Récupère une partie du jeu d'entrainement et des labels
		inputs, outputs								= digitImport.getTrainingSet(length = rc_TrainingLength)
		inputs_test, outputs_test					= digitImport.getTestSet(length = rc_TestLength)
		
		# Informations
		#print "trainingSetLength : {}".format(rc_TrainingLength)
		#print "Jeu d'entraînement contenant {} entrées de taille {} pour un total de {} digit(s) soumis".format(inputs.shape[0], inputs.shape[1],inputs.shape[0]/rc_ImagesSize)
		#print "Jeu de test contenant {} entrées de taille {} pour un total de {} digit(s) soumis".format(inputs_test.shape[0], inputs_test.shape[1], inputs_test.shape[0]/rc_ImagesSize)
		
		# Données
		data = [None, [(inputs, outputs)], None]
		
		# Entrainement du réseau
		#print "Entraînement du réseau de neurones...."
		flow.train(data)
		
		# Applique le réseau entraîné au jeu de test
		testout = flow(inputs_test)

		# Digit error rate
		der = digitImport.digitErrorRate(testout[0])
		
		# Affiche
		#print "Nombre de digits mal classifiés : {}".format(der)
		
		print der
		results[n] = der
	
	print "Average : {}".format(np.average(results))
	print "Sigma : {}".format(np.std(results))
