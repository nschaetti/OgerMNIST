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
rc_SpectralRadius 			= 0.2				# Spectral radius
rc_Size 					= 1000				# Taille du réservoir
rc_InputScaling 			= 0.5				# Dimensionnement des entrées
rc_LeakRate					= 0					# Leak rate
rc_nbDigits					= 10				# Nombre de digits (sorties)
rc_TrainingLength			= 60000				# Longueur d'entrainement
rc_TestLength				= 10000				# Longeur de test
rc_ImagesSize				= 15				# Taille des digits
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
	digitImport = MNISTImporter.Open(sys.argv[1])
	
	# Nouveau RNN
	reservoir = DigitReservoir(digitImport, training_length = rc_TrainingLength, test_length = rc_TestLength, nb_digit = rc_nbDigits)
	
	# Construit le réseau
	reservoir.createRNN(spectral_radius = rc_SpectralRadius, input_scaling = rc_InputScaling, reservoir_size = rc_Size, select_method = rc_SelectMethod, dtype = 'float32')
		
	# Entraîne le réseau
	reservoir.Train()
		
	# Test le réseau
	[der, out] = reservoir.Test()
	
	print "Digit Error Rate : {}".format(der)
