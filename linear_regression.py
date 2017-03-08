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
# On fait une simple régression linéaire sur les digits
#
#########################################################################

####################################################
# Fonction principale
####################################################
if __name__ == "__main__":
	
	# Charge
	digitImport = MNISTImporter()
	digitImport.Load(sys.argv[1])
	
	# Normalize les images
	digitImport.resizeImages(size = 20)
	
	# Applis les images
	digitImport.flattenImages()
	
	# Un simple noeud de classification et un classifieur
	readout = Oger.nodes.RidgeRegressionNode(input_dim = 400, output_dim = 10, dtype='float64')
	classifier = SingleInputClassifier(input_dim = 10, output_dim = 1)
	
	# Récupère une partie du jeu d'entrainement et des labels
	inputs = digitImport.trainImages
	inputs_test = digitImport.testImages
	outputs, outputs_test = digitImport.generateShortLabels()
	
	# Informations sur les données
	print "Informations sur les données : "
	print "Inputs : " + str(inputs.shape)
	print "Outputs : " + str(outputs.shape)
	print "Inputs test : " + str(inputs_test.shape)
	print "Outputs test : " + str(outputs_test.shape)
	
	# Données
	data = [[(inputs, outputs)], None]
	
	# Construction du flux
	flow = mdp.Flow([readout, classifier], verbose=1)
	
	# Entrainement du réseau
	flow.train(data)

	# Applique le réseau entraîné au jeu de test
	testout = flow(inputs_test)

	# Digit error rate
	print "Taux d'erreur : " + str(float(digitImport.digitErrorRate(testout)))
