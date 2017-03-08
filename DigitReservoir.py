#!/usr/bin/env python
# -*- coding: utf-8 -*-

##########################################################################
#
# Auteur : Nils Schaetti <nils.schaetti@univ-comte.fr>
# Date : 19.04.2015 18:19:57 17:59:05
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
import DigitReservoir
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm

#
# CLASS DigitReservoir
# Classe représentant un réservoir simple de reconnaissance MNIST
#
class DigitReservoir:
	
	##############################################################
	# Constructeur
	##############################################################
	def __init__(self, mnistImporter, training_length, test_length, nb_digit = 10):
		""" Constructeur, on initialize les paramètres généraux du réservoir """
		
		# Paramètres
		print "DigitReservoir : {} digits d'entraînement, {} digits de test".format(training_length, test_length)
		self.digitImport = mnistImporter
		self.trainingSetLength = training_length
		self.testSetLength = test_length
		self.nbDigit = nb_digit
		
	##############################################################
	# Crée le réseau
	##############################################################
	def createRNN(self, spectral_radius = 0.9, input_scaling = 0.5, reservoir_size = 100, select_method = "average", dtype='float64'):
		""" On crée le reservoir et le reste du réseau selon un rayon spectral, une taille,
		    un facteur d'échelle d'entrée spécifiés """
		
		# Paramètres
		self.spectralRadius = spectral_radius
		self.inputScaling = input_scaling
		self.reservoirSize = reservoir_size
		self.selectMethod = select_method
		
		# Construction des noeuds
		print "Création d'un réservoir de taille {}, avec {} sortie(s) et {} entrée(s)!".format(reservoir_size, self.nbDigit, self.digitImport.nbInputs, dtype = dtype)
		self.reservoir = Oger.nodes.ReservoirNode(input_dim = self.digitImport.nbInputs, output_dim = reservoir_size, input_scaling = input_scaling, spectral_radius = spectral_radius)
		self.readout = Oger.nodes.RidgeRegressionNode(output_dim = self.nbDigit, dtype = dtype)
		self.classifier = DigitClassifierNode(mnist_space = self.digitImport.interImagesSpace, label_space_ratio = self.digitImport.interImagesRatio, digit_space_ratio = self.digitImport.digitImageRatio, image_size = self.digitImport.imagesSize, nb_digit = self.nbDigit, method = self.selectMethod, input_dim = self.nbDigit, dtype = dtype)
		
		# Construction du flux
		print "Construction du flux..."
		self.flow = mdp.Flow([self.reservoir, self.readout, self.classifier], verbose=1)
		Oger.utils.make_inspectable(Oger.nodes.ReservoirNode)
		
	##############################################################
	# Entraîne le réseau
	##############################################################
	def Train(self):
		""" Entraîne le réseau, on récupère les données, on construit 
		    un jeu pour chaque partie du réseau, puis on le soumet
		    au réseau afin qu'il l'apprenne """
		
		# Récupère une partie du jeu d'entrainement et des labels
		inputs, outputs								= self.digitImport.getTrainingSet(length = self.trainingSetLength)
		self.inputs_test, self.outputs_test			= self.digitImport.getTestSet(length = self.testSetLength)
		
		# Informations
		print "trainingSetLength : {}".format(self.trainingSetLength)
		print "Jeu d'entraînement contenant {} entrées de taille {} pour un total de {} digit(s) soumis".format(inputs.shape[0], inputs.shape[1],inputs.shape[0]/self.digitImport.imagesSize)
		print "Jeu de test contenant {} entrées de taille {} pour un total de {} digit(s) soumis".format(self.inputs_test.shape[0], self.inputs_test.shape[1], self.inputs_test.shape[0]/self.digitImport.imagesSize)
		
		# Données
		data = [None, [(inputs, outputs)], None]
		
		# Entrainement du réseau
		print "Entraînement du réseau de neurones...."
		self.flow.train(data)

	##############################################################
	# Test le réseau
	##############################################################
	def Test(self):
		""" On teste le réseau en comptant le nombre de digit mal
		    classé """
		
		# Applique le réseau entraîné au jeu de test
		testout = self.flow(self.inputs_test)

		# Digit error rate
		return self.digitImport.digitErrorRate(testout[0]), testout[1]
