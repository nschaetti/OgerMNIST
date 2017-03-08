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
import MySQLdb as mdb

#########################################################################
#
# Ici on recherche des paramètres optimaux et on écrit les 
# résultats dans une base MySQL
#
#########################################################################

# Paramètres du réservoir
rc_SpectralRadius 			= mdp.numx.arange(0.01, 1.5, 0.05)			# Spectral radius (12)
rc_Size 					= mdp.numx.arange(300,801,50)				# Taille du réservoir (5)
rc_InputScaling 			= mdp.numx.arange(0.01,4,0.25)				# Dimensionnement des entrées (5)
rc_LeakRate					= mdp.numx.arange(0.01,1.01,0.1)			# Leak rate (10)
rc_Bias						= mdp.numx.arange(0, 1.01, 0.2)				# Biais (5)
rc_nbDigits					= 10										# Nombre de digits (sorties)
rc_TrainingLength			= 60000										# Longueur d'entrainement
rc_TestLength				= 10000										# Longeur de test
rc_ImagesSize				= mdp.numx.arange(10, 26, 5)				# Taille des digits

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
	
	# Output
	con = mdb.connect('localhost','root','snow2458--','thesis')
	cur = con.cursor()
	
	# Toutes les tailles
	for size in rc_Size:
		# Spactral radius
		for spectral_radius in rc_SpectralRadius:
			# Input scaling
			for input_scaling in rc_InputScaling:
				
				# Vérifie si déjà fait
				cur.execute("SELECT * FROM analyze_sr_is_size WHERE size = {} AND spectral_radius = {} AND input_scaling = {}".format(size,spectral_radius,input_scaling))
				rows = cur.fetchall()
			
				if len(rows) == 0:
					# Calcul
					count = 0.0
					for i in range(3):
						# Créer du réservoir
						#reservoir = Oger.nodes.LeakyReservoirNode(input_dim = digitImport.nbInputs, output_dim = size, input_scaling = input_scaling, spectral_radius = spectral_radius, leak_rate = leak_rate, bias_scaling = bias, dtype='float64')
						reservoir = Oger.nodes.LeakyReservoirNode(input_dim = digitImport.nbInputs, output_dim = size, input_scaling = input_scaling, spectral_radius = spectral_radius, dtype='float64')
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
						count += float(digitImport.digitErrorRate(testout,with_miss_array = False, per_digit = False, pos_table = False))
						
					# Inscrit les résultats
					insert = "INSERT INTO analyze_sr_is_size (`id`, `size`, `spectral_radius`, `input_scaling`, `digit_error_rate`) VALUES (NULL, '{}', '{}', '{}', '{}');".format(size,spectral_radius,input_scaling,count/3.0)
					print "Insertion... {}".format(insert)
					cur.execute(insert)
					con.commit()
				else:
					print "OK INSERT INTO analyze_sr_is_size (`id`, `size`, `spectral_radius`, `input_scaling`, `digit_error_rate`) VALUES (NULL, '{}', '{}', '{}',);".format(size,spectral_radius,input_scaling)
	
	# Ferme la connexion
	con.close()	

