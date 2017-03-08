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
rc_LeakRate					= [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]					# Leak rate
rc_InputSparsity			= 0.9
rc_ReservoirSparsity		= 0.1
rc_Biais					= 1.0

rc_Size 					= 100				# Taille du réservoir
rc_nbDigits					= 10				# Nombre de digits (sorties)
rc_TrainingLength			= 60000				# Longueur d'entrainement
rc_TestLength				= 10000				# Longeur de test
rc_ImagesSize				= [8,10,12,14,16,18,20,22,24,26,28]				# Taille des digits
rc_SelectMethod 			= "average"			# Methode d'élection
rc_SampleSize				= 60

####################################################
# Fonction principale
####################################################
if __name__ == "__main__":
	
	# Vérifie les params
	if len(sys.argv) != 2:
		print "Donnez un nom de fichier!"
		exit()
		
	for image_size in rc_ImagesSize:
		print "Paramètre rc_ImagesSize valant {}".format(image_size)
		
		# Import les digits
		print "Importation des digits depuis {}".format(sys.argv[1])
		digitImport = MNISTImporter()
		digitImport.Load(sys.argv[1])
		
		# Resize image
		digitImport.resizeImages(image_size)
		
		# Ajouteu une série temporelle
		digitImport.addTimeserie()
		
		# Génère les labels
		digitImport.generateLabels()
		
		# Construction des noeuds
		for leakyrate in rc_LeakRate:
			results = np.zeros(rc_SampleSize)
			print "Paramètre rc_LeakRate valant {}".format(leakyrate)
			for n in np.arange(0,rc_SampleSize):
				#print "n = {}".format(n)
				#print "Création d'un réservoir de taille {}, avec {} sortie(s) et {} entrée(s)!".format(rc_Size, rc_nbDigits, rc_ImagesSize, dtype = 'float64')
				reservoir = Oger.nodes.LeakyReservoirNode(leak_rate = leakyrate, input_dim = image_size, output_dim = rc_Size, input_scaling = rc_InputScaling, spectral_radius = rc_SpectralRadius, bias_scaling=rc_Biais, sparsity = rc_InputSparsity, w_sparsity = rc_ReservoirSparsity)
				readout = Oger.nodes.RidgeRegressionNode(output_dim = rc_nbDigits, dtype = 'float64')
				classifier = DigitClassifierNode(mnist_space = digitImport.interImagesSpace, label_space_ratio = digitImport.interImagesRatio, digit_space_ratio = digitImport.digitImageRatio, image_size = image_size, nb_digit = rc_nbDigits, method = 'average', input_dim = rc_nbDigits, dtype = 'float64')
				
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
				results[n] = der
				
			print '\033[93m' + "Paramètre rc_LeakRate valant {} : moyenne de {}".format(leakyrate, np.average(results)) + '\033[0m'
		print ""
