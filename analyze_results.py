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
# Ici on analyze les résultats d'un réservoir
#
#########################################################################

# Paramètres du réservoir
rc_SpectralRadius 			= 0.9				# Spectral radius
rc_Size 					= 100				# Taille du réservoir
rc_InputScaling 			= 1.0				# Dimensionnement des entrées
rc_LeakRate					= 0					# Leak rate
rc_nbDigits					= 10				# Nombre de digits (sorties)
rc_TrainingLength			= 60000				# Longueur d'entrainement
rc_TestLength				= 10000				# Longeur de test
rc_ImagesSize				= 15				# Taille des digits
rc_SelectMethod 			= "average"			# Methode d'élection
rc_DigitToAnalyze			= 2					# Digit à analyser
rc_LengthToAnalyze			= 40				# Longueur à analyser

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
	digitImport = MNISTImporter.Open(sys.argv[1])
	
	# Créer du réservoir
	print "Création d'un réservoir de taille {}, avec {} sortie(s) et {} entrée(s)!".format(rc_Size, rc_nbDigits, digitImport.nbInputs)
	reservoir = Oger.nodes.ReservoirNode(input_dim = digitImport.nbInputs, output_dim = rc_Size, input_scaling = rc_InputScaling, spectral_radius = rc_SpectralRadius, dtype='float64')
	readout = Oger.nodes.RidgeRegressionNode(output_dim = rc_nbDigits, dtype='float64')
	classifier = DigitClassifierNode(mnist_space = digitImport.interImagesSpace, label_space_ratio = digitImport.interImagesRatio, digit_space_ratio = digitImport.digitImageRatio, image_size = digitImport.imagesSize, nb_digit = rc_nbDigits, method = rc_SelectMethod, input_dim = rc_nbDigits, analyze_sampling = rc_LengthToAnalyze, dtype='float64')
	
	# Récupère une partie du jeu d'entrainement et des labels
	inputs, outputs						= digitImport.getTrainingSet(length = rc_TrainingLength)
	inputs_test, outputs_test			= digitImport.getTestSet(length = rc_TestLength)
	data = [None, [(inputs, outputs)], None]
	
	# Construction du flux
	print "Construction du flux..."
	flow = mdp.Flow([reservoir, readout, classifier], verbose=1)
	Oger.utils.make_inspectable(Oger.nodes.ReservoirNode)
	
	# Entrainement du réseau
	print "Entraînement du réseau de neurones...."
	flow.train(data)
	
	## Applique le réseau entraîné au jeu de test
	testout, out = flow(inputs_test)

	err = MNISTImporter.digitErrorRate(testout, digitImport.testLabels)
	print "Digit error rate : {}".format(err)

	## Digit error rate
	#res, per_digit = digitImport.digitErrorRate(testout, per_digit = True)
	#print "Digit Error Rate : {}".format(res)
	
	## Affiche
	#fig = plt.figure()
	#ax = fig.add_subplot(1,1,1)
	#major_ticks = np.arange(0, rc_LengthToAnalyze * digitImport.entrySize, digitImport.entrySize)
	##minor_ticks = np.arange(0, rc_LengthToAnalyze * digitImport.entrySize, digitImport.entrySize/2.0)
	#ax.set_xticks(major_ticks)                                                       
	#ax.set_yticks(major_ticks)                                                       
	#ax.grid(which='both')   
	#plt.plot(np.arange(0,rc_LengthToAnalyze * digitImport.entrySize), out[:,rc_DigitToAnalyze], 'v-')
	#plt.plot(np.arange(0,rc_LengthToAnalyze * digitImport.entrySize), out[:,rc_DigitToAnalyze+3], 'o-')
	#plt.plot(np.arange(0,rc_LengthToAnalyze * digitImport.entrySize), digitImport.getTestSet(start = 0, length = rc_LengthToAnalyze)[1][:,rc_DigitToAnalyze])
	#plt.show()
	
	## Digit Error Rate
	#print "Digit Error Rate : {}".format(res)
	#print "Digit Error Rate per digit : "
	#for i in range(10):
		#print "{} : {} %".format(i,per_digit[i])
