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
rc_Size 					= 100				# Taille du réservoir
rc_InputScaling 			= 0.2				# Dimensionnement des entrées
rc_LeakRate					= 0.2				# Leak rate
rc_Bias						= 0.4
rc_nbDigits					= 10				# Nombre de digits (sorties)
rc_TrainingLength			= 60000				# Longueur d'entrainement
rc_TestLength				= 10000				# Longeur de test
rc_ImagesSize				= 22				# Taille des digits
rc_SelectMethod 			= "average"			# Methode d'élection
rc_Epochs					= 1					# Epoch
rc_FirstLayerSize			= 100				# Taille premier niveau
rc_SecondLayerSize			= 200				# Taille second niveau

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
	
	# Série normale
	digitImport.resetImages()
	digitImport.resizeImages(size = rc_ImagesSize)
	digitImport.addTimeserie()
	
	# Série normale
	digitImport.rotateImages(angle = 30)
	digitImport.resizeImages(size = rc_ImagesSize)
	digitImport.addTimeserie()
	
	# Série normale
	digitImport.rotateImages(angle = 60)
	digitImport.resizeImages(size = rc_ImagesSize)
	digitImport.addTimeserie()
	
	# Ajoute les labels
	digitImport.generateLabels()
	
	# Génère les labels
	outputs, outputs_test = digitImport.generateShortLabels()
	
	# Informations
	print "Label généré..."
	print "Longueur du jeu d'entraînement : " + str(digitImport.trainSetLength)
	print "Nombre d'entrées : " + str(digitImport.nbInputs)
	print "Longueur d'une entrée : " + str(digitImport.entrySize)
	
	# Réservoir et jointure
	reservoir = Oger.nodes.LeakyReservoirNode(input_dim = digitImport.nbInputs, output_dim = rc_Size, input_scaling = rc_InputScaling, spectral_radius = rc_SpectralRadius, leak_rate = rc_LeakRate, bias_scaling = rc_Bias)
	joiner = JoinedStatesNode(image_size = digitImport.entrySize, input_dim = rc_Size)
	
	# Récupère une partie du jeu d'entrainement et des labels
	inputs, out					= digitImport.getTrainingSet(length = rc_TrainingLength)
	inputs_test, out_test		= digitImport.getTestSet(length = rc_TestLength)
	
	# Informations
	print "Génération des états du réservoir..."
	print "Outputs : " + str(outputs.shape)
	print "Outputs : " + str(outputs.ndim)
	print "Inputs : " + str(inputs.shape)
	print "Inputs : " + str(inputs.ndim)

	# Calcule les états
	tmp_states = reservoir.execute(inputs)
	states = joiner.execute(tmp_states)
	
	# Informations
	print "Etats générés... forme : " + str(states.shape)
	
	# Generate a small subset of the data.
	data_size = rc_Size * rc_ImagesSize
	rbmnode1 = Oger.nodes.ERBMNode(data_size, rc_FirstLayerSize)
	rbmnode2 = Oger.nodes.ERBMNode(rc_FirstLayerSize, rc_SecondLayerSize)
	percnode = Oger.nodes.PerceptronNode(rc_SecondLayerSize, 10, transfer_func=Oger.utils.SoftmaxFunction)
	
	# Greedy pretraining of RBMs
	print 'Training first layer...'
	for epoch in range(rc_Epochs):
		for c in mdp.utils.progressinfo(states):
			rbmnode1.train(c.reshape((1, data_size)), n_updates=1, epsilon=.1)
	
	hiddens = rbmnode1(states)
	
	print 'Training second layer...'
	for epoch in range(rc_Epochs):
		for c in mdp.utils.progressinfo(hiddens):
			rbmnode2.train(c.reshape((1, rc_FirstLayerSize)), n_updates=1, epsilon=.1)
	
	# Store weights.
	myflow = rbmnode1 + rbmnode2 + percnode
	bpnode = Oger.gradient.BackpropNode(myflow, Oger.gradient.GradientDescentTrainer(momentum=.9), loss_func=Oger.utils.ce)

	# Fine-tune for classification
	print 'Fine-tuning for classification...'
	for epoch in range(rc_Epochs):
		for i in mdp.utils.progressinfo(range(len(states))):
			label = np.array(np.eye(10)[digitImport.trainLabels[i], :])
			bpnode.train(x=states[i].reshape((1, data_size)), t=label.reshape((1, 10)))
	
	# Calcule les états
	test_states = joiner.execute(reservoir.execute(inputs_test))
	
	# Evaluate performance on test set.
	testout = bpnode(test_states)
	
	# Evaluation
	testout[np.arange(testout.shape[0]), np.argmax(testout, axis=1)] = 1
	testout[testout < 1] = 0
	t_test = np.array([int(i) for i in digitImport.testLabels])
	correct = np.sum(testout[np.arange(len(t_test)), t_test])
	print "{} bien classé(s) sur {}".format(correct,len(digitImport.testLabels))
	print '\033[93m' + 'Proportion of correctly classified digits:', correct / float(len(digitImport.testLabels)), '\033[0m'
	
