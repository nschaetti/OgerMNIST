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
rc_SpectralRadius 			= mdp.numx.arange(0.01, 1.2, 0.1)			# Spectral radius (12)
rc_Size 					= mdp.numx.arange(100,501,100)				# Taille du réservoir (5)
rc_InputScaling 			= mdp.numx.arange(0.01,2.5,0.5)				# Dimensionnement des entrées (5)
rc_LeakRate					= mdp.numx.arange(0.01,1.01,0.1)			# Leak rate (10)
rc_Bias						= mdp.numx.arange(0, 1.01, 0.2)				# Biais (5)
rc_nbDigits					= 10										# Nombre de digits (sorties)
rc_TrainingLength			= 60000										# Longueur d'entrainement
rc_TestLength				= 10000										# Longeur de test
rc_ImagesSize				= 15										# Taille des digits

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
	#digitImport = MNISTImporter.Open(sys.argv[1])
	
	digitImport.Load(sys.argv[1])
	digitImport.resizeImages(size = rc_ImagesSize)
	digitImport.addTimeserie()
	
	#digitImport.resetImages()
	digitImport.resizeImages(size = rc_ImagesSize)
	digitImport.rotateImages(angle = 30)
	digitImport.addTimeserie()
	
	#digitImport.resetImages()
	digitImport.resizeImages(size = rc_ImagesSize)
	digitImport.rotateImages(angle = 60)
	digitImport.addTimeserie()
	
	digitImport.generateLabels()
	digitImport.showTrainingSet()
	
	# Génère les labels
	outputs, outputs_test = digitImport.generateShortLabels()
	
	# Output
	con = mdb.connect('localhost','root','snow2458--','thesis')
	cur = con.cursor()
	
	# Toutes les tailles
	for size in rc_Size:
		# Spactral radius
		for spectral_radius in rc_SpectralRadius:
			# Input scaling
			for input_scaling in rc_InputScaling:
				for leak_rate in rc_LeakRate:
					for bias in rc_Bias:
						
						# Vérifie si déjà fait
						cur.execute("SELECT * FROM analyze_sr_is_size_lr_joined WHERE size = {} AND spectral_radius = {} AND input_scaling = {} AND leak_rate = {} AND bias = {}".format(size,spectral_radius,input_scaling,leak_rate,bias))
						rows = cur.fetchall()
						
						if len(rows) == 0:
							count = 0.0
							for i in range(3):
								
								# Réservoir et jointure
								reservoir = Oger.nodes.LeakyReservoirNode(input_dim = digitImport.nbInputs, output_dim = size, input_scaling = input_scaling, spectral_radius = spectral_radius, leak_rate = leak_rate, bias_scaling = bias)
								joiner = JoinedStatesNode(image_size = rc_ImagesSize, input_dim = size)
								
								# Regression et classifieur
								readout = Oger.nodes.RidgeRegressionNode(output_dim = rc_nbDigits, dtype='float64')
								classifier = DigitClassifierNode(mnist_space = 0, label_space_ratio = 0, digit_space_ratio = 0, image_size = 1, nb_digit = rc_nbDigits, method = "average", input_dim = rc_nbDigits, dtype='float64')
								
								# Récupère une partie du jeu d'entrainement et des labels
								inputs, out										= digitImport.getTrainingSet(length = rc_TrainingLength)
								inputs_test, out_test							= digitImport.getTestSet(length = rc_TestLength)

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
								
								# Applique le réseau entraîné au jeu de test
								testout, out = flow(test_states)
								
								# Digit error rate
								count += float(digitImport.digitErrorRate(testout,with_miss_array = False, per_digit = False, pos_table = False))
								
							# Inscrit les résultats
							insert = "INSERT INTO analyze_sr_is_size_lr_joined (`id`, `size`, `spectral_radius`, `input_scaling`, `leak_rate`, `bias`, `digit_error_rate`) VALUES (NULL, '{}', '{}', '{}', '{}', '{}', '{}');".format(size,spectral_radius,input_scaling,leak_rate,bias,count/3.0)
							print "Insertion... {}".format(insert)
							cur.execute(insert)
							con.commit()
						else:
							print "OK : INSERT INTO analyze_sr_is_size_lr_joined (`id`, `size`, `spectral_radius`, `input_scaling`, `leak_rate`, `bias`, `digit_error_rate`) VALUES (NULL, '{}', '{}', '{}', '{}', '{}');".format(size,spectral_radius,input_scaling,leak_rate,bias)
	
	# Ferme la connexion
	con.close()	


