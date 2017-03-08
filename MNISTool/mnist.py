#!/usr/bin/env python
# -*- coding: utf-8 -*-

##########################################################################
#
# Auteur : Nils Schaetti <nils.schaetti@edu.univ-comte.fr>
# Date : 19.04.2015 17:59:05
# Lieu : Nyon, Suisse
#
# Importation et traitement des digits MNIST, application de filtres
# et de déformations, transformation en série temporelle.
#
###########################################################################

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import numpy as np
import Oger
import pylab as pyl
import mdp
import os
import cPickle
import struct
import sys
from skimage.transform import resize, rotate, integral_image
from skimage import measure
from skimage.feature import corner_harris, corner_subpix, corner_peaks, hog
from skimage import feature, segmentation, color
from skimage.transform import warp, AffineTransform, PiecewiseAffineTransform, warp
from skimage.draw import ellipse
from skimage.filter.rank import entropy
from skimage.filter import sobel
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from skimage import color
from skimage.util.shape import view_as_windows
from skimage.util.montage import montage2d
from scipy.cluster.vq import kmeans2
from scipy import ndimage as ndi
import math
import scipy as sci

trainImagesIndex = "trainimages"								# Index des images d'entrainement
testImagesIndex = "testimages"									# Index des images de test
trainLabelsIndex = "trainlabels"								# Index des labels d'entrainement
testLabelsIndex = "testlabels"									# Index des labels de test

#
# CLASS MNISTImporter
# Classe permettant l'importation des digits MNIST, leur traitement, leur conversion en
# série temporelle et de multiple transformations avant de le passer au réservoir
#
class MNISTImporter:

	##############################################################
	# Constructeur
	##############################################################
	def __init__(self, image_size = 28, train_length = 60000, test_length = 10000, mnist_space = 0, label_space_ratio = 0, digit_space_ratio = 0, wrong_value = -1, right_value = 1):
		""" Constructeur, on initialize les variables par défaut """
		
		self.imagesSize = image_size								# Taille des images
		self.Loaded = False											# Données chargées
		self.trainSetLength = train_length							# Longueur jeu de learning
		self.testSetLength = test_length							# Longueur jeu de test
		self.trainImages = np.zeros((train_length,28,28))			# Images d'apprentissage
		self.testImages = np.zeros((test_length,28,28))				# Images de tests
		self.originalTrainImages = np.zeros((train_length,28,28))	# Images d'apprentissage originales
		self.oroginalTestImages = np.zeros((test_length,28,28))		# Images de tests originales
		self.trainLabels = np.zeros(train_length)					# Labels d'apprentissage
		self.testLabels = np.zeros(test_length)						# Labels de tests
		self.Convert = False										# Converti en séries temporelles?
		self.nbInputs = 0											# Nombre d'entrées pour le réseau de neurones
		self.interImagesSpace = mnist_space							#
		self.interImagesRatio = label_space_ratio					#
		self.digitImageRatio = digit_space_ratio					#
		self.entrySize = self.imagesSize + mnist_space				# Taille totale d'un digit
		self.nbTS = 0												# Nombre de série temporelle
		self.wrongValue = wrong_value								# Sortie pour un digit non reconnu
		self.rightValue = right_value								# Sortie pour un digit reconnu
		
	##############################################################
	# Charge les données MNIST
	##############################################################
	def Load(self, mnist_in):
		""" Ici on charge les MNIST depuis un fichier de class Python """
		
		# Charge
		data = cPickle.load(open(mnist_in))

		# Images et labels
		self.trainImages = data[trainImagesIndex]
		self.originalTrainImages = data[trainImagesIndex]
		self.trainLabels = data[trainLabelsIndex]
		self.testImages = data[testImagesIndex]
		self.originalTestImages = data[testImagesIndex]
		self.testLabels = data[testLabelsIndex]
		
		# Reforme
		self.trainImages.shape = (self.trainSetLength, self.imagesSize, self.imagesSize)
		self.testImages.shape =  (self.testSetLength, self.imagesSize, self.imagesSize)
		
		# Echelle unitaire
		self.trainImages /= 255.0
		self.testImages /= 255.0
		
		# Chargé
		self.Loaded = True
		
	##############################################################
	# Ajoute une suite d'images mélangés au jeu d'entrainement
	##############################################################
	def addMixedDigits(self):
		""" Ici on va ajouter une suite de digit pris dans le jeu
		    d'entraînement mais mélangés afin de rendre les digits
		    indépendants """
		
		# Taille entrainement
		train_length = self.trainSetLength
		    
		# Tableau des digits deja ajoutés
		check_added = np.array([False]*train_length)
		
		# Jusqu'à ce que tous les digits soient ajoutés
		while np.bincount(check_added)[0] != 0:
			
			# Index d'un digit au hasard
			rand_index = np.random.randint(train_length, size=1)[0]
			
			# Si il n'est pas encore ajouté
			if check_added[rand_index] == False:
				
				# Ajoute l'image
				self.trainImages = np.vstack((self.trainImages,np.array([self.trainImages[rand_index]])))
				
				# Ajoute le label
				self.trainLabels = np.append(self.trainLabels,self.trainLabels[rand_index])
				
				# Traité
				check_added[rand_index] = True
				
				# Longueur
				self.trainSetLength += 1
				
				print "1 image ajoutée, longueur totale : " + str(self.trainSetLength)
				
	##############################################################
	# Ajoute une suite d'images avec transformations affines aléatoire
	##############################################################
	def addAffineImages(self, start = 0, length = 60000, scale_dev = 0.2, rotation_dev = 0.5, shear_dev = 0.3, verbose = 0):
		
		# Ajoute chaque digits
		for i in np.arange(start, start+length):
			
			# Ajoute l'image
			ok = False
			while ok == False:
				
				# Paramètres
				scale = (1.0+round(np.random.rand(),4)*(scale_dev*2)-scale_dev, 1.0+round(np.random.rand(),4)*(scale_dev*2)-scale_dev)
				rotation = round(np.random.rand(),4)*(rotation_dev*2)-rotation_dev
				shear = (round(np.random.rand(),4)*(shear_dev*2)-shear_dev,round(np.random.rand(),4)*(shear_dev*2)-shear_dev)
				
				try:
					self.trainImages = np.vstack((self.trainImages,np.array([self.affineTransformImage(self.trainImages[i,:,:], scale = scale, rotation = rotation, shear = shear)])))
					ok = True
				except sci.spatial.qhull.QhullError:
					print "sci.spatial.qhull.QhullError! " + str(i)
					ok = False
					pass
			
			# Ajoute le label
			self.trainLabels = np.append(self.trainLabels,self.trainLabels[i])
			
			# Longueur
			self.trainSetLength  += 1
			
			if verbose == 1:
				if i % 100 == 0:
					print "1 image affine ajoutée, longueur total : " + str(self.trainSetLength)
			
	##############################################################
	# Ajoute une suite d'images avec transformations élastiques
	##############################################################
	def addElasticImages(self, start = 0, length = 60000, scale = 6.0, sigma = 1.1, verbose = 0):
		
		# Ajoute chaque digits
		for i in np.arange(start, start+length):
			
			# Ajoute l'image
			ok = False
			while ok == False:
				try:
					self.trainImages = np.vstack((self.trainImages,np.array([self.elasticTransformImage(self.trainImages[i,:,:], scale = scale, sigma = sigma)])))
					ok = True
				except sci.spatial.qhull.QhullError:
					print "sci.spatial.qhull.QhullError! " + str(i)
					ok = False
					pass
			
			# Ajoute le label
			self.trainLabels = np.append(self.trainLabels,self.trainLabels[i])
			
			# Longueur
			self.trainSetLength  += 1
			
			if verbose == 1:
				if i % 100 == 0:
					print "1 image élastique ajoutée, longueur total : " + str(self.trainSetLength)
					
	##############################################################
	# Ajoute une suite d'images floutées
	##############################################################
	def addBlurredImages(self, start = 0, length = 60000, sigma = 1):
		
		# Ajoute chaque digits
		for i in np.arange(start, start+length):
			
			# Ajoute l'image
			ok = False
			while ok == False:
				try:
					self.trainImages = np.vstack((self.trainImages,np.array([self.blurImage(self.trainImages[i,:,:], sigma = sigma)])))
					ok = True
				except sci.spatial.qhull.QhullError:
					print "sci.spatial.qhull.QhullError! " + str(i)
					ok = False
					pass
			
			# Ajoute le label
			self.trainLabels = np.append(self.trainLabels,self.trainLabels[i])
			
			# Longueur
			self.trainSetLength  += 1
			
			if verbose == 1:
				if i % 100 == 0:
					print "1 image floutée ajoutée, longueur total : " + str(self.trainSetLength)
					
	##############################################################
	# Ajoute une suite d'images bruitées
	##############################################################
	def addNoisyImages(self, start = 0, length = 60000, sigma = 0.05):
		
		# Ajoute chaque digits
		for i in np.arange(start, start+length):
			
			# Ajoute l'image
			ok = False
			while ok == False:
				try:
					self.trainImages = np.vstack((self.trainImages,np.array([self.noiseImage(self.trainImages[i,:,:], sigma = sigma)])))
					ok = True
				except sci.spatial.qhull.QhullError:
					print "sci.spatial.qhull.QhullError! " + str(i)
					ok = False
					pass
			
			# Ajoute le label
			self.trainLabels = np.append(self.trainLabels,self.trainLabels[i])
			
			# Longueur
			self.trainSetLength  += 1
			
			if verbose == 1:
				if i % 100 == 0:
					print "1 image floutée bruitée, longueur total : " + str(self.trainSetLength)
		
	##############################################################
	# Réinitialize les digits
	##############################################################
	def resetImages(self):
		self.trainImages = self.originalTrainImages
		self.testImages = self.originalTestImages
		self.imagesSize = self.trainImages.shape[1]
		self.entrySize = self.imagesSize + self.interImagesSpace
		
	##############################################################
	# Applatis les images
	##############################################################
	def flattenImages(self):
		self.trainImages.shape = (self.trainSetLength, self.imagesSize * self.imagesSize)
		self.testImages.shape = (self.testSetLength, self.imagesSize * self.imagesSize)
		
	##############################################################
	# Ajoute une série temporelle
	##############################################################
	def addTimeserie(self):
		""" Ici on ajoute une série temporelle crée à partir des images """
		
		# Séries temporelles
		trainTS = np.zeros((self.trainSetLength * self.entrySize, self.imagesSize))
		testTS = np.zeros((self.testSetLength * self.entrySize, self.imagesSize))
		
		# Changement de forme
		train_digits = self.trainImages.copy()
		test_digits = self.testImages.copy()
		
		# Converti les images d'apprentissage
		for i in np.arange(0, self.trainSetLength):
			for j in np.arange(0, self.imagesSize):
				trainTS[i*self.entrySize+j,:] = train_digits[i,:,j]
					
		# Converti les images d'apprentissage
		for i in np.arange(0, self.testSetLength):
			for j in np.arange(0, self.imagesSize):
				testTS[i*self.entrySize+j,:] = test_digits[i,:,j]
					
		# Ajoute la série
		try:
			self.trainTSImages = np.hstack((self.trainTSImages, trainTS.copy()))
			self.testTSImages = np.hstack((self.testTSImages, testTS.copy()))
		except AttributeError:
			self.trainTSImages = trainTS.copy()
			self.testTSImages = testTS.copy()
					
		# Nombre de séries
		self.nbTS += 1
					
		# Nombre d'entrées
		self.nbInputs = self.trainTSImages.shape[1]
			
		# Converti
		self.Convert = True
		
	##############################################################
	# Ajoute une série temporelle composé d'ondelette
	##############################################################
	def addTimeserieOndelette(self, start = 4, end = 20, nbsteps = 7, axis = None):
		
		# Séries temporelles
		trainTS = np.zeros((self.trainSetLength * nbsteps, end * end))
		testTS = np.zeros((self.testSetLength * nbsteps, end * end))
		
		# Entrée
		self.entrySize = nbsteps
		
		# Converti les images d'apprentissage
		for i in np.arange(0, self.trainSetLength):
			t = 0
			for s in np.linspace(start,end,nbsteps):
				
				# Arrondi
				size = int(s)
				
				# Resizée
				if axis == None:
					sized = resize(self.trainImages[i,:,:],(size,size))
				elif axis == 0:
					sized = resize(self.trainImages[i,:,:],(size,end))
				elif axis == 1:
					sized = resize(self.trainImages[i,:,:],(end,size))
				sized = resize(sized,(end,end))
				sized.shape = (end * end)
				
				# Image
				trainTS[i*self.entrySize+t,:] = sized
				
				# Next
				t += 1
				
		# Converti les images de test
		for i in np.arange(0, self.testSetLength):
			t = 0
			for s in np.linspace(start,end,nbsteps):
				
				# Arrondi
				size = int(s)
				
				# Resizée
				# Resizée
				if axis == None:
					sized = resize(self.testImages[i,:,:],(size,size))
				elif axis == 0:
					sized = resize(self.testImages[i,:,:],(size,end))
				elif axis == 1:
					sized = resize(self.testImages[i,:,:],(end,size))
				sized = resize(sized,(end,end))
				sized.shape = (end * end)
				
				# Image
				testTS[i*self.entrySize+t,:] = sized
				
				# Next
				t += 1
		
		# Ajoute la série
		try:
			self.trainTSImages = np.hstack((self.trainTSImages, trainTS))
			self.testTSImages = np.hstack((self.testTSImages, testTS))
		except AttributeError:
			self.trainTSImages = trainTS
			self.testTSImages = testTS
		
		# Nombre de séries
		self.nbTS += 1
		
		# Nombre d'entrées
		self.nbInputs = end * end
		
		# Converti
		self.Convert = True
	
	##############################################################
	# Ajoute une série temporelle composé d'images répétées
	##############################################################
	def addTimeserieRepeat(self, nbsteps = 7):
		
		# Taille
		size = int(self.trainImages.shape[1])
		
		# Séries temporelles
		trainTS = np.zeros((self.trainSetLength * nbsteps, size * size))
		testTS = np.zeros((self.testSetLength * nbsteps, size * size))
		
		# Entrée
		self.entrySize = nbsteps

		# Converti les images d'apprentissage
		for i in np.arange(0, self.trainSetLength):
			for s in np.arange(0,nbsteps):
				
				# Image
				finale = self.trainImages[i,:,:].copy()
				finale.shape = (size * size)
				trainTS[i*self.entrySize+s,:] = finale
				
		# Converti les images de test
		for i in np.arange(0, self.testSetLength):
			for s in np.arange(0,nbsteps):
				
				# Image
				finale = self.testImages[i,:,:].copy()
				finale.shape = (size * size)
				testTS[i*self.entrySize+s,:] = finale
		
		# Ajoute la série
		try:
			self.trainTSImages = np.hstack((self.trainTSImages, trainTS))
			self.testTSImages = np.hstack((self.testTSImages, testTS))
		except AttributeError:
			self.trainTSImages = trainTS
			self.testTSImages = testTS
		
		# Nombre de séries
		self.nbTS += 1
		
		# Nombre d'entrées
		self.nbInputs = size * size
		
		# Converti
		self.Convert = True
	
	##############################################################
	# Ajoute une série temporelle composé d'images répétées
	##############################################################
	def addTimeserieOne(self, nbsteps = 7):
		
		# Taille
		size = int(self.trainImages.shape[1])
		
		# Séries temporelles
		trainTS = np.zeros((self.trainSetLength * nbsteps, size * size))
		testTS = np.zeros((self.testSetLength * nbsteps, size * size))
		
		# Entrée
		self.entrySize = nbsteps
		
		# Converti les images d'apprentissage
		for i in np.arange(0, self.trainSetLength):
			t = 0
			for s in np.arange(0,nbsteps):
				
				# Image
				if s == 0:
					finale = self.trainImages[i,:,:].copy()
					finale.shape = (size * size)
					trainTS[i*self.entrySize+t,:] = finale
				else:
					trainTS[i*self.entrySize+t,:] = np.zeros(size*size)
				
				# Next
				t += 1
				
		# Converti les images de test
		for i in np.arange(0, self.testSetLength):
			t = 0
			for s in np.arange(0,nbsteps):
				
				# Image
				if s == 0:
					finale = self.testImages[i,:,:].copy()
					finale.shape = (size * size)
					testTS[i*self.entrySize+t,:] = finale
				else:
					testTS[i*self.entrySize+t,:] = np.zeros(size*size)
				
				# Next
				t += 1
		
		# Ajoute la série
		try:
			self.trainTSImages = np.hstack((self.trainTSImages, trainTS))
			self.testTSImages = np.hstack((self.testTSImages, testTS))
		except AttributeError:
			self.trainTSImages = trainTS
			self.testTSImages = testTS
		
		# Nombre de séries
		self.nbTS += 1
		
		# Nombre d'entrées
		self.nbInputs = size * size
		
		# Converti
		self.Convert = True
	
	##############################################################
	# Ajoute une série temporelle composé d'ondelette
	##############################################################
	def addTimeserieRandomOndelette(self, start = 4, end = 20, nbsteps = 7, axis = None):
		
		# Séries temporelles
		trainTS = np.zeros((self.trainSetLength * nbsteps, end * end))
		testTS = np.zeros((self.testSetLength * nbsteps, end * end))
		
		# Entrée
		self.entrySize = nbsteps
		
		# Converti les images d'apprentissage
		for i in np.arange(0, self.trainSetLength):
			t = 0
			for s in np.linspace(start,end,nbsteps):
				
				# Arrondi
				size = int(np.random.randint(start,end))
				
				# Resizée
				if axis == None:
					sized = resize(self.trainImages[i,:,:],(size,size))
				elif axis == 0:
					sized = resize(self.trainImages[i,:,:],(size,end))
				elif axis == 1:
					sized = resize(self.trainImages[i,:,:],(end,size))
				sized = resize(sized,(end,end))
				sized.shape = (end * end)
				
				# Image
				trainTS[i*self.entrySize+t,:] = sized
				
				# Next
				t += 1
				
		# Converti les images de test
		for i in np.arange(0, self.testSetLength):
			t = 0
			for s in np.linspace(start,end,nbsteps):
				
				# Arrondi
				size = int(np.random.randint(start,end))
				
				# Resizée
				# Resizée
				if axis == None:
					sized = resize(self.testImages[i,:,:],(size,size))
				elif axis == 0:
					sized = resize(self.testImages[i,:,:],(size,end))
				elif axis == 1:
					sized = resize(self.testImages[i,:,:],(end,size))
				sized = resize(sized,(end,end))
				sized.shape = (end * end)
				
				# Image
				testTS[i*self.entrySize+t,:] = sized
				
				# Next
				t += 1
		
		# Ajoute la série
		try:
			self.trainTSImages = np.hstack((self.trainTSImages, trainTS))
			self.testTSImages = np.hstack((self.testTSImages, testTS))
		except AttributeError:
			self.trainTSImages = trainTS
			self.testTSImages = testTS
		
		# Nombre de séries
		self.nbTS += 1
		
		# Nombre d'entrées
		self.nbInputs = end * end
		
		# Converti
		self.Convert = True
	
	
	##############################################################
	# Ajoute une série temporelle gaussienne
	##############################################################
	def addTimeserieGaussian(self, start = 4, end = 0, nbsteps = 7):
		
		# Séries temporelles
		trainTS = np.zeros((self.trainSetLength * nbsteps, self.imagesSize*self.imagesSize))
		testTS = np.zeros((self.testSetLength * nbsteps, self.imagesSize*self.imagesSize))
		
		# Entrée
		self.entrySize = nbsteps
		
		# Converti les images d'apprentissage
		for i in np.arange(0, self.trainSetLength):
			t = 0
			for sigma in np.linspace(start,end,nbsteps):
				
				# Convolution avec un noyau gaussien
				blurred = sci.ndimage.filters.gaussian_filter(self.trainImages[i,:,:], sigma = sigma)
				blurred.shape = (self.imagesSize * self.imagesSize)
				
				# Image
				trainTS[i*self.entrySize+t,:] = blurred
				
				# Next
				t += 1
		
		# Converti les images de test
		for i in np.arange(0, self.testSetLength):
			t = 0
			for sigma in np.linspace(start,end,nbsteps):
				
				# Convolution avec un noyau gaussien
				blurred = sci.ndimage.filters.gaussian_filter(self.testImages[i,:,:], sigma = sigma)
				blurred.shape = (self.imagesSize * self.imagesSize)
				
				# Image
				testTS[i*self.entrySize+t,:] = blurred
				
				# Next
				t += 1
		
		# Ajoute la série
		try:
			self.trainTSImages = np.hstack((self.trainTSImages, trainTS))
			self.testTSImages = np.hstack((self.testTSImages, testTS))
		except AttributeError:
			self.trainTSImages = trainTS
			self.testTSImages = testTS
		
		# Nombre de séries
		self.nbTS += 1
		
		# Nombre d'entrées
		self.nbInputs = self.imagesSize*self.imagesSize
		
		# Converti
		self.Convert = True
	
	##############################################################
	# Ajoute une série temporelle DoG
	##############################################################
	def addTimeserieDoG(self, length = 7, sigma_step = 2):
		
		# Séries temporelles
		trainTS = np.zeros((self.trainSetLength * length, self.imagesSize*self.imagesSize))
		testTS = np.zeros((self.testSetLength * length, self.imagesSize*self.imagesSize))
		
		# Entrée
		self.entrySize = length
		
		# Converti les images d'apprentissage
		for i in np.arange(0, self.trainSetLength):
			t = 0
			for l in np.fliplr([np.arange(1,length+1)])[0]:
				
				# Premier pas
				blurred1 = sci.ndimage.filters.gaussian_filter(self.trainImages[i,:,:], sigma = sigma_step*(l-1))
				
				# Deuxième pas
				blurred2 = sci.ndimage.filters.gaussian_filter(self.trainImages[i,:,:], sigma = sigma_step*l)
				
				# Différence
				dog = blurred2 - blurred1
				
				# Forme
				dog.shape = (self.imagesSize * self.imagesSize)
				
				# Image
				trainTS[i*self.entrySize+t,:] = dog
				
				# Next
				t += 1
		
		# Converti les images de test
		for i in np.arange(0, self.testSetLength):
			t = 0
			for l in np.fliplr([np.arange(1,length+1)])[0]:
				
				# Premier pas
				blurred1 = sci.ndimage.filters.gaussian_filter(self.testImages[i,:,:], sigma = sigma_step*(l-1))
				
				# Deuxième pas
				blurred2 = sci.ndimage.filters.gaussian_filter(self.testImages[i,:,:], sigma = sigma_step*l)
				
				# Différence
				dog = blurred2 - blurred1
				
				# Forme
				dog.shape = (self.imagesSize * self.imagesSize)
				
				# Image
				testTS[i*self.entrySize+t,:] = dog
				
				# Next
				t += 1
		
		# Ajoute la série
		try:
			self.trainTSImages = np.hstack((self.trainTSImages, trainTS.copy()))
			self.testTSImages = np.hstack((self.testTSImages, testTS.copy()))
		except AttributeError:
			self.trainTSImages = trainTS.copy()
			self.testTSImages = testTS.copy()
		
		# Nombre de séries
		self.nbTS += 1
		
		# Nombre d'entrées
		self.nbInputs = self.imagesSize*self.imagesSize
		
		# Converti
		self.Convert = True
	
	##############################################################
	# Génère les labels courts
	##############################################################
	def generateShortLabels(self, length = None):
		""" Génère les labels courts """
		
		if length == None:
			length = self.trainSetLength
		
		# Tableaux pour les labels
		trainLabels = np.zeros((length, 10)) + self.wrongValue
		testLabels = np.zeros((self.testSetLength, 10)) + self.wrongValue
		
		# Inscrit les résultats d'entraînement attendus
		for i in np.arange(0, length):
			current_label = self.trainLabels[i]
			trainLabels[i, int(current_label)] = self.rightValue
		
		# Inscrit les résultats de test attendus
		for i in np.arange(0, self.testSetLength):
			current_label = self.testLabels[i]
			testLabels[i, int(current_label)] = self.rightValue
			
		return (trainLabels, testLabels)
		
	##############################################################
	# Génère les résultats attendus
	##############################################################
	def generateLabels(self):
		""" Génère la série de labels """
		
		pos = 0
		
		# Paramère
		mnist_space = self.interImagesSpace
		label_space_ratio = self.interImagesRatio
		
		# Tableaux pour les labels
		self.trainTSLabels = np.zeros((self.entrySize * self.trainSetLength, 10)) + self.wrongValue
		self.testTSLabels = np.zeros((self.entrySize * self.testSetLength, 10)) + self.wrongValue
		
		# Inscrit les résultats d'entrainement attendus
		for i in np.arange(0, self.trainSetLength):
			current_label = self.trainLabels[i]
			self.trainTSLabels[pos + int(self.imagesSize * self.digitImageRatio) : pos + self.imagesSize + int(mnist_space * label_space_ratio), int(current_label)] = np.repeat(self.rightValue, int(self.imagesSize * (1 - self.digitImageRatio)) + int(mnist_space * label_space_ratio))
			pos += self.entrySize
			
		# Inscrit les résultats de test attendus
		pos = 0
		for i in np.arange(0, self.testSetLength):
			current_label = self.testLabels[i]
			self.testTSLabels[pos + int(self.imagesSize * self.digitImageRatio) : pos + self.imagesSize + int(mnist_space * label_space_ratio), int(current_label)] = np.repeat(self.rightValue, int(self.imagesSize * (1 - self.digitImageRatio)) + round(mnist_space * label_space_ratio))
			pos += self.entrySize
			
	##############################################################
	# Génère les résultats attendus
	##############################################################
	def generateImageLabels(self):
		""" Génère la série de labels """
		
		# Tableaux pour les labels
		self.trainTSLabels = np.zeros((self.entrySize * self.trainSetLength, 10)) + self.wrongValue
		self.testTSLabels = np.zeros((self.entrySize * self.testSetLength, 10)) + self.wrongValue
		
		# Inscrit les résultats d'entrainement attendus
		for i in np.arange(0, self.trainSetLength):
			current_label = self.trainLabels[i]
			self.trainTSLabels[i * self.entrySize : (i+1) * self.entrySize, int(current_label)] = np.repeat(self.rightValue, self.entrySize)
			
		# Inscrit les résultats de test attendus
		for i in np.arange(0, self.testSetLength):
			current_label = self.testLabels[i]
			self.testTSLabels[i * self.entrySize : (i+1) * self.entrySize, int(current_label)] = np.repeat(self.rightValue, self.entrySize)
			
	##############################################################
	# Génère les résultats attendus sur un seul label
	##############################################################
	def generateLabel(self, multiply = 1.0):
		""" Génère une série de labels """
		
		pos = 0
		
		# Paramère
		mnist_space = self.interImagesSpace
		label_space_ratio = self.interImagesRatio
		
		# Tableaux pour les labels
		self.trainTSLabels = np.zeros((self.entrySize * self.trainSetLength,1)) - 1
		self.testTSLabels = np.zeros((self.entrySize * self.testSetLength,1)) - 1
		
		# Inscrit les résultats d'entrainement attendus
		for i in np.arange(0, self.trainSetLength):
			current_label = self.trainLabels[i]
			self.trainTSLabels[pos + int(self.imagesSize * self.digitImageRatio) : pos + self.imagesSize + int(mnist_space * label_space_ratio),0] = np.repeat((current_label/10.0)*multiply, int(self.imagesSize * (1 - self.digitImageRatio)) + int(mnist_space * label_space_ratio))
			pos += self.entrySize
			
		# Inscrit les résultats de test attendus
		pos = 0
		for i in np.arange(0, self.testSetLength):
			current_label = self.testLabels[i]
			self.testTSLabels[pos + int(self.imagesSize * self.digitImageRatio) : pos + self.imagesSize + int(mnist_space * label_space_ratio),0] = np.repeat((current_label/10.0)*multiply, int(self.imagesSize * (1 - self.digitImageRatio)) + round(mnist_space * label_space_ratio))
			pos += self.entrySize
		
	##############################################################
	# Applique la détection de bord Sobel à toutes les images
	##############################################################
	def applySobel(self):
		""" Applique la détection de bords à toutes les images chargées """
		
		# Applique à chaque images
		for i in np.arange(0,self.trainSetLength):
			self.trainImages[i,:,:] = sobel(self.trainImages[i,:,:])
			
		for i in np.arange(0,self.testSetLength):
			self.testImages[i,:,:] = sobel(self.testImages[i,:,:])
			
	##############################################################
	# Applique la détection de bords Sobel à toute une série temp
	##############################################################
	def applySobelToTimeserie(self, TS=0):
		""" Applique la détection de bords de Sobel à toute une 
		    séries temporelles """
		
		self.trainTSImages[:,TS*self.imagesSize:TS*self.imagesSize+self.imagesSize] = sobel(self.trainTSImages[:,TS*self.imagesSize:TS*self.imagesSize+self.imagesSize])
		self.testTSImages[:,TS*self.imagesSize:TS*self.imagesSize+self.imagesSize] = sobel(self.testTSImages[:,TS*self.imagesSize:TS*self.imagesSize+self.imagesSize])
	
	##############################################################
	# Applique une transformation ondellete sur les images
	##############################################################
	def applyOndelette(self, size, axis = None):
		
		# Jeu d'entraînement
		for i in np.arange(0, self.trainSetLength):
			
			# Resizée
			if axis == None:
				sized = resize(self.trainImages[i,:,:],(size,size))
			elif axis == 0:
				sized = resize(self.trainImages[i,:,:],(size,self.imagesSize))
			elif axis == 1:
				sized = resize(self.trainImages[i,:,:],(self.imagesSize,size))
			sized = resize(sized,(self.imagesSize,self.imagesSize))
			self.trainImages[i,:,:] = sized
		
		# Jeu de test
		for i in np.arange(0, self.testSetLength):
			
			# Resizée
			if axis == None:
				sized = resize(self.testImages[i,:,:],(size,size))
			elif axis == 0:
				sized = resize(self.testImages[i,:,:],(size,self.imagesSize))
			elif axis == 1:
				sized = resize(self.testImages[i,:,:],(self.imagesSize,size))
			sized = resize(sized,(self.imagesSize,self.imagesSize))
			self.testImages[i,:,:] = sized
	
	##############################################################
	# Applique des transformations affines aléatoires aux images
	##############################################################
	def applyRandomAffineTransformations(self, scale_dev = 0.2, rotation_dev = 3, shear_dev = 0.7):
		""" On applique des transformations affines aléatoire
		    aux images avec des déviations spécifiés dans les 
		    arguments """
		
		# Jeu d'entrainement
		print "Application au jeu d'entraînement..."
		for i in np.arange(0, self.trainSetLength):
			ok = False
			while ok == False:
				if i % 100 == 0:
					print "# Image train " + str(i)
			
				# Paramètres
				scale = (1.0+round(np.random.rand(),4)*(scale_dev*2)-scale_dev, 1.0+round(np.random.rand(),4)*(scale_dev*2)-scale_dev)
				rotation = round(np.random.rand(),4)*(rotation_dev*2)-rotation_dev
				shear = (round(np.random.rand(),4)*(shear_dev*2)-shear_dev,round(np.random.rand(),4)*(shear_dev*2)-shear_dev)
				
				# Transform
				try:
					self.trainImages[i,:,:] = self.affineTransformImage(self.trainImages[i,:,:], scale = scale, rotation = rotation, shear = shear)
					ok = True
				except sci.spatial.qhull.QhullError:
					print "sci.spatial.qhull.QhullError! " + str(i)
					ok = False
					pass
		
		# Jeu de test
		print "Application au jeu de test..."
		for i in np.arange(0, self.testSetLength):
			ok = False
			while ok == False:
				if i % 100 == 0:
					print "# Image test " + str(i)
				
				# Paramètres
				scale = (1.0+round(np.random.rand(),4)*(scale_dev*2)-scale_dev, 1.0+round(np.random.rand(),4)*(scale_dev*2)-scale_dev)
				rotation = round(np.random.rand(),4)*(rotation_dev*2)-rotation_dev
				shear = (round(np.random.rand(),4)*(shear_dev*2)-shear_dev,round(np.random.rand(),4)*(shear_dev*2)-shear_dev)
				
				# Transform
				try:
					self.testImages[i,:,:] = self.affineTransformImage(self.testImages[i,:,:], scale = scale, rotation = rotation, shear = shear)
					ok = True
				except sci.spatial.qhull.QhullError:
					print "sci.spatial.qhull.QhullError! " + str(i)
					ok = False
					pass
					
	##############################################################
	# Applique des transformations élastiques aléatoires aux images
	##############################################################
	def applyElasticTransformations(self, scale = 6.0, sigma = 1.1):
		""" On applique des transformations élastiques aléatoires
		    aux images avec un facteur de taille et une déviation
		    standard spécifiés """
		    
		# Jeu d'entraînement
		print "Application au jeu d'entraînement..."
		for i in np.arange(0, self.trainSetLength):
			ok = False
			while ok == False:
				if i % 100 == 0:
					print "# Image train " + str(i)
					
				# Transform
				try:
					self.trainImages[i,:,:] = self.elasticTransformImage(self.trainImages[i,:,:], scale = scale, sigma = sigma)
					ok = True
				except sci.spatial.qhull.QhullError:
					print "sci.spatial.qhull.QhullError! " + str(i)
					ok = False
					pass
					
		# Jeu de test
		print "Application au jeu de test..."
		for i in np.arange(0, self.testSetLength):
			ok = False
			while ok == False:
				if i % 100 == 0:
					print "# Image test " + str(i)
					
				# Transform
				try:
					self.testImages[i,:,:] = self.elasticTransformImage(self.testImages[i,:,:], scale = scale, sigma = sigma)
					ok = True
				except sci.spatial.qhull.QhullError:
					print "sci.spatial.qhull.QhullError! " + str(i)
					ok = False
					pass
		
	##############################################################
	# Sérialize les données dans un fichier
	##############################################################
	def Save(self, mnist_out):
		""" On enregistre toutes les données dans une fichier pour 
		    pouvoir les réutiliser par la suite sans avoir à les 
		    retraiter """
		
		if self.Loaded == True and self.Convert == True:
			data = dict()
			data["originalTrainImages"] = self.trainImages
			data["originalTrainLabels"] = self.trainLabels
			data["originalTestImages"] = self.testImages
			data["originalTestLabels"] = self.testLabels
			data["trainImages"] = self.trainTSImages
			data["trainLabels"] = self.trainTSLabels
			data["testImages"] = self.testTSImages
			data["testLabels"] = self.testTSLabels
			cPickle.dump(self, open(mnist_out, 'w'), protocol=2)
		elif self.Loaded == True:
			data = dict()
			data["originalTrainImages"] = self.trainImages
			data["originalTrainLabels"] = self.trainLabels
			data["originalTestImages"] = self.testImages
			data["originalTestLabels"] = self.testLabels
			cPickle.dump(self, open(mnist_out, 'w'), protocol=2)
	
	##############################################################
	# Ouvre un fichier contenant des digits en séries temporelles
	##############################################################
	@staticmethod
	def Open(file_in):
		""" On charge les données depuis un fichier qu'on a créé 
		    auparavant avec cette même classe """
		
		# Charge
		importer = cPickle.load(open(file_in))
		importer.originalTrainImages = importer.trainImages.copy()
		importer.originalTrainLabels = importer.trainLabels.copy()
		importer.originalTestImages = importer.testImages.copy()
		importer.originalTestLabels = importer.testLabels.copy()
		
		return importer
		
	##############################################################
	# Affiche les digits d'entrainement
	##############################################################
	def showTrainingSet(self, start = 0, length = 20):
		""" On affiche le jeu d'entrainement avec une posittion de 
		    départ et une longueur de digits à afficher """
		
		if self.Loaded == True and self.Convert == True:
			if self.trainTSLabels.shape[1] == 10:
				imgplot = plt.imshow(np.hstack((self.trainTSImages[start * self.entrySize : start * self.entrySize + length * self.entrySize,:], self.trainTSLabels[start * self.entrySize : start * self.entrySize + length * self.entrySize, : ] / 2.0 + 0.5)), cmap = cm.Greys_r)
				plt.show()
			elif self.trainTSLabels.shape[1] == 1:
				imgplot = plt.imshow(np.hstack((self.trainTSImages[start * self.entrySize : start * self.entrySize + length * self.entrySize,:], self.trainTSLabels[start * self.entrySize : start * self.entrySize + length * self.entrySize, : ])), cmap = cm.Greys_r)
				plt.show()
				
	##############################################################
	# Affiche les digits d'entraînement
	##############################################################
	def showTrainingSetImage(self, start = 0, length = 20):
		
		# Pour chaque image
		for i in np.arange(start, start + length):
			# Pour chaque entrée
			for j in np.arange(0,self.entrySize):
				# Image
				im = self.trainTSImages[i*self.entrySize+j,:]
				im.shape = (int(np.sqrt(im.shape[0])),int(np.sqrt(im.shape[0])))
				imgplot = plt.imshow(im, cmap = cm.Greys_r)
				plt.show()
				
	##############################################################
	# Affiche les digits de test
	##############################################################
	def showTestSetImage(self, start = 0, length = 20):
		
		# Pour chaque image
		for i in np.arange(start, start + length):
			# Pour chaque entrée
			for j in np.arange(0,self.entrySize):
				# Image
				im = self.testTSImages[i*self.entrySize+j,:]
				im.shape = (int(np.sqrt(im.shape[0])),int(np.sqrt(im.shape[0])))
				imgplot = plt.imshow(im, cmap = cm.Greys_r)
				plt.show()
	
	##############################################################
	# Affiche les digits de test
	##############################################################
	def showTestSet(self, start = 0, length = 20):
		""" On affiche le jeu de test avec une posittion de 
		    départ et une longueur de digits à afficher """
		
		if self.Loaded == True and self.Convert == True:
			if self.testTSLabels.shape[1] == 10:
				imgplot = plt.imshow(np.hstack((self.testTSImages[start * self.entrySize : start * self.entrySize + length * self.entrySize], self.testTSLabels[start * self.entrySize : start * self.entrySize + length * self.entrySize, : ] / 2.0 + 0.5)), cmap = cm.Greys_r)
				plt.show()
			elif self.testTSLabels.shape[1] == 1:
				imgplot = plt.imshow(np.hstack((self.testTSImages[start * self.entrySize : start * self.entrySize + length * self.entrySize], self.testTSLabels[start * self.entrySize : start * self.entrySize + length * self.entrySize, : ])), cmap = cm.Greys_r)
				plt.show()
			
	##############################################################
	# Affiche une image
	##############################################################
	def showImage(self, index = 0, sset = "train"):
		""" Affiche une image grâce à imshow en niveau de gris """

		if sset == "train":
			imgplot = plt.imshow(self.trainImages[index,:,:], cmap = cm.Greys_r)
		else:
			imgplot = plt.imshow(self.testImages[index,:,:], cmap = cm.Greys_r)
		plt.show()
		
	##############################################################
	# Affiche une image avec une transformation élastique
	##############################################################
	def showImageWithElasticTransform(self, index = 0, sset = "train", scaling_factor = 1.0, standard_deviation = 1.0, show_field = 0):
		""" Affiche une image en lui appliquant une transformation
		    élsatique caractérisée par une déviation standard
		    d'un noyau gaussien et d'un facteur d'agrandissement """
		    
		# Choisi l'image
		if sset == "train":
			im = self.trainImages[index,:,:]
		else:
			im = self.testImages[index,:,:]
		
		# Taille
		rows, cols = im.shape[0], im.shape[1]
		
		# Grille
		src_cols = np.linspace(2, cols-2, 14)
		src_rows = np.linspace(2, rows-2, 14)
		src_rows, src_cols = np.meshgrid(src_rows, src_cols)
		src = np.dstack([src_cols.flat, src_rows.flat])[0]
		
		# Random deplacement fields
		dep_x = (np.random.rand(14,14)*2.0)-1.0
		dep_y = (np.random.rand(14,14)*2.0)-1.0
		
		if show_field == 1:
			Q = pyl.quiver(src_cols, src_rows, dep_x, dep_y, pivot='mid')
			qk = pyl.quiverkey(Q, 0.9, 1.05, 1, "", fontproperties={'weight' : 'bold'})
			plt.plot(src_cols, src_rows, 'k.')
			plt.axis([-1, 29, -1, 29])
			plt.title("pivot='mid'; overy third arrow; units='inches'")
			plt.show()
		
		# Convolution avec un noyau gaussien
		conv_x = sci.ndimage.filters.gaussian_filter(dep_x, sigma = standard_deviation)
		conv_y = sci.ndimage.filters.gaussian_filter(dep_y, sigma = standard_deviation)
		
		if show_field == 1:
			Q = pyl.quiver(src_cols, src_rows, conv_x, conv_y, pivot='mid')
			qk = pyl.quiverkey(Q, 0.9, 1.05, 1, "", fontproperties={'weight' : 'bold'})
			plt.plot(src_cols, src_rows, 'k.')
			plt.axis([-1, 29, -1, 29])
			plt.title("pivot='mid'; overy third arrow; units='inches'")
			plt.show()
		
		# Multiplication par un facteur d'échelle
		conv_x *= scaling_factor
		conv_y *= scaling_factor
		
		if show_field == 1:
			Q = pyl.quiver(src_cols, src_rows, conv_x, conv_y, pivot='mid')
			qk = pyl.quiverkey(Q, 0.9, 1.05, 1, "", fontproperties={'weight' : 'bold'})
			plt.plot(src_cols, src_rows, 'k.')
			plt.axis([-1, 29, -1, 29])
			plt.title("pivot='mid'; overy third arrow; units='inches'")
			plt.show()
		
		# Application
		dst = src.copy()
		for x in np.arange(0,14):
			for y in np.arange(0,14):
				dst[x*14+y,0] += conv_x[x,y]
				dst[x*14+y,1] += conv_y[x,y]
		
		# Piecewise affine transform
		tform = PiecewiseAffineTransform()
		tform.estimate(dst, src)
		
		# Output image
		out_rows = im.shape[0]
		out_cols = im.shape[1]
		out = warp(im, tform, output_shape=(out_rows, out_cols))

		# Affiche
		both = np.vstack((im,out))
		imgplot = plt.imshow(both, cmap = cm.Greys_r)
		plt.show()
		
	##############################################################
	# Affiche une image avec une transformation affine
	##############################################################
	def showImageWithAffineTransform(self, index = 0, sset = "train", scale = (1,1), rotation = 0, shear = (0,0)):
		""" Affiche une image en lui appliquant une transformation
		    affine définie par l'aggrandissement, la rotation et 
		    le shear """
		   
		# Choisi l'image
		if sset == "train":
			im = self.trainImages[index,:,:]
		else:
			im = self.testImages[index,:,:]
		
		# Centre de l'image
		centerX = im.shape[1]/2.0
		centerY = im.shape[0]/2.0
		
		# Matrice de transformation
		m_trans = np.matrix([[scale[0], shear[0]],[shear[1], scale[1]]])
		
		# Taille
		rows, cols = im.shape[0], im.shape[1]
		
		# Grille
		src_cols = np.linspace(0, cols, 14)
		src_rows = np.linspace(0, rows, 14)
		src_rows, src_cols = np.meshgrid(src_rows, src_cols)
		src = np.dstack([src_cols.flat, src_rows.flat])[0]
		
		# Apply transformation matrix
		dst = src.copy()
		for i in range(0,src.shape[0]):
			src_pos = src[i,:] - np.array([centerX,centerY])
			trans = np.array(m_trans.dot(src_pos).T)
			trans.shape = 2
			trans += np.array([centerX,centerY])
			dst[i,:] = trans
		
		# Piecewise affine transform
		tform = PiecewiseAffineTransform()
		tform.estimate(dst, src)
		
		# Output image
		out_rows = im.shape[0]
		out_cols = im.shape[1]
		out = warp(im, tform, output_shape=(out_rows, out_cols))
		
		# Tourne l'image
		out = rotate(out, angle = rotation)

		# Affiche
		both = np.vstack((im,out))
		imgplot = plt.imshow(both, cmap = cm.Greys_r)
		plt.show()
		
	##############################################################
	# Affiche une image avec la détection de contours
	##############################################################
	def showImageWithContourFinding(self, index = 0, sset = "train"):
		""" Affiche une image en niveau de gris avec les contours 
		    détecté """
		
		# Choisi l'image
		if sset == "train":
			im = self.trainImages[index,:,:]
		else:
			im = self.testImages[index,:,:]
		
		# Trouve les contours
		contours = measure.find_contours(im,0.9)
		
		# Affiche l'image et les contours
		fig, ax = plt.subplots()
		ax.imshow(im, interpolation='nearest', cmap=plt.cm.gray)
		print contours
		for n, contour in enumerate(contours):
			ax.plot(contour[:,1], contour[:,0], linewidth=2)
		
		# Affiche le tout
		plt.show()
		
	##############################################################
	# Affiche une image avec la détection de coins
	##############################################################
	def showImageWithCornerFindig(self, index = 0, sset = "train"):
		""" Affiche une image en niveau de gris avec les coins 
		    détectés """
		
		# Choisi l'image
		if sset == "train":
			im = self.trainImages[index,:,:]
		else:
			im = self.testImages[index,:,:]
			
		# Corners
		coords = corner_peaks(corner_harris(im), min_distance=5)
		coords_subpix = corner_subpix(im, coords, window_size=13)
		
		# Affiche
		fig, ax = plt.subplots()
		ax.imshow(im, interpolation="nearest", cmap=plt.cm.gray)
		ax.plot(coords[:,1], coords[:,0], '.b', markersize=3)
		ax.plot(coords_subpix[:,1], coords_subpix[:,0], '+r', markersize=15)
		plt.show()
		
	##############################################################
	# Applique le filtre de Gabor à une image
	##############################################################
	def showImageWithGaborFilter(self, index = 0, sset = "train"):
		
		# Choisi l'image
		if sset == "train":
			im = self.trainImages[index,:,:]
		else:
			im = self.testImages[index,:,:]
			
		patch_shape = 8, 8
		n_filters = 49
		patches1 = view_as_windows(im, patch_shape)
		patches1 = patches1.reshape(-1, patch_shape[0] * patch_shape[1])[::8]
		fb1, _ = kmeans2(patches1, n_filters, minit = 'points')
		fb1 = fb1.reshape((-1,) + patch_shape)
		fb1_montage = montage2d(fb1, rescale_intensity = True) 
		
		# Affiche
		fig, ax = plt.subplots()
		ax.imshow(fb1_montage, interpolation="nearest", cmap=plt.cm.gray)
		plt.show()
		
	##############################################################
	# Affiche une image flotées
	##############################################################
	def showBlurredImage(self, index = 0, sset = "train", sigma = 1.0):
		
		# Choisi l'image
		if sset == "train":
			im = self.trainImages[index,:,:]
		else:
			im = self.testImages[index,:,:]
		
		# Floute l'image
		blurred = sci.ndimage.filters.gaussian_filter(im, sigma = sigma)
		
		both = np.vstack((im,blurred))
		plt.imshow(both, cmap=plt.cm.gray)
		plt.show()
		
	##############################################################
	# Affiche l'entropie d'une image
	##############################################################
	def showImageEntropy(self, index = 0, sset = "train"):
		""" Montre l'entropie d'une image """
		
		# Choisi l'image
		if sset == "train":
			im = self.trainImages[index,:,:]
		else:
			im = self.testImages[index,:,:]
		
		plt.imshow(entropy(im, disk(5)), cmap=plt.cm.gray)
		plt.show()
		
	##############################################################
	# Affiche une image bruitée
	##############################################################
	def showNoisyImage(self, index = 0, sset = "train", sigma = 1.0):
		
		# Choisi l'image
		if sset == "train":
			im = self.trainImages[index,:,:]
		else:
			im = self.testImages[index,:,:]
		
		# Taille
		width = im.shape[1]
		height = im.shape[0]
		
		# Ajoute du bruit
		im += np.random.normal(loc = 0.0, scale = sigma, size = (height, width))
		
		plt.imshow(im, cmap=plt.cm.gray)
		plt.show()
		
	##############################################################
	# Affiche une image avec la détection de bords
	##############################################################
	def showImageWithEdgesDetection(self, index = 0, sigma = 3, sset = "train"):
		""" Montre une image avec la détection des bords """
		
		# Choisi l'image
		if sset == "train":
			im = self.trainImages[index,:,:]
		else:
			im = self.testImages[index,:,:]
			
		# Détection des bords
		edges = sobel(im)
		
		# Affiche
		plt.imshow(edges, cmap=plt.cm.gray)
		plt.show()
		
	##############################################################
	# Affiche une image et son historigramme d'orientation
	##############################################################
	def showImageWithOrientation(self, index = 0, sset = "train"):
		""" Montre les historigrame d'orientation d'une image """
		
		# Choisi l'image
		if sset == "train":
			im = self.trainImages[index,:,:]
		else:
			im = self.testImages[index,:,:]
			
		# Orientation
		fd, hog_image = hog(im, orientations=8, pixels_per_cell=(3,3), cells_per_block=(2,2), visualise=True)
		
		# Affiche
		plt.imshow(hog_image, cmap=plt.cm.gray)
		plt.show()
		
	##############################################################
	# Affiche une image normalizée
	##############################################################
	def showDenoisedImage(self, index = 0, sset = "train"):
		""" Montre une image après la supression des bruits """
		
		# Choisi l'image
		if sset == "train":
			im = self.trainImages[index,:,:]
		else:
			im = self.testImages[index,:,:]
		
		# Segmentation
		denoise = denoise_tv_chambolle(im, weight=0.1, multichannel=False)
		
		# Affiche
		plt.imshow(denoise)
		plt.show()
		
	##############################################################
	# Affiche les ondelettes d'une image
	##############################################################
	def showImageOndelette(self, index = 0, sset = "train", start = 4, end = 28, nbsteps = 5):
		
		# Choisi l'image
		if sset == "train":
			im = self.trainImages[index,:,:]
		else:
			im = self.testImages[index,:,:]
		
		# Pour chaque taille
		for i in np.linspace(start,end,nbsteps):
			
			# Arrondi
			size = round(i)
			
			# Resizée
			sized = resize(im,(size,size))
			
			# Affiche
			plt.imshow(resize(sized,(im.shape[0],im.shape[1])), cmap = cm.Greys_r)
			plt.show()
			
	##############################################################
	# Affiche la pyramide gaussienne d'une image
	##############################################################
	def showImageGaussianPyramid(self, index = 0, sset = "train", start = 4, end = 0, nbsteps = 7):
		
		# Choisi l'image
		if sset == "train":
			im = self.trainImages[index,:,:]
		else:
			im = self.testImages[index,:,:]
		
		# Pour chaque taille
		for i in np.linspace(start,end,nbsteps):
			
			# Sigma
			print "Sigma : " + str(i)
			
			# Convolution avec un noyau gaussien
			blurred = sci.ndimage.filters.gaussian_filter(im, sigma = i)
			
			# Affiche
			plt.imshow(blurred, cmap = cm.Greys_r)
			plt.show()
			
	##############################################################
	# Affiche la différence de gaussien d'une image
	##############################################################
	def showImageDoG(self, index = 0, sset = "train", length = 7, sigma_step = 2):
		
		# Choisi l'image
		if sset == "train":
			im = self.trainImages[index,:,:]
		else:
			im = self.testImages[index,:,:]
		
		# Image de base
		blurred = im.copy()
		
		# Pour chaque taille
		for i in range(length):
			
			# Blur
			new_blurred = sci.ndimage.filters.gaussian_filter(blurred, sigma = sigma_step)
			
			# Affiche
			plt.imshow(new_blurred - blurred, cmap = cm.Greys_r)
			plt.show()
			
			blurred = new_blurred
			
	##############################################################
	# Retourne la taille réelle d'une image
	##############################################################
	def getRealImageSize(self, index = 0):
		""" On calcule la taille d'une image sans les bords noirs """
		
		# Compteurs
		line_count = 0
		column_count = 0
		
		# L'image
		image = self.trainImages[index,:,:]
		
		# Ligne et colonnes à zero
		lines = np.all(image == 0, axis = 1)
		columns = np.all(image == 0, axis = 0)
		
		# Compte les éléments
		for i in range(self.imagesSize):
			if lines[i] == False:
				line_count += 1
			if columns[i] == False:
				column_count += 1
				
		return line_count, column_count
			
	##############################################################
	# Retourne la taille moyenne des images
	##############################################################
	def getAverageImageSize(self, sset = "train"):
		""" On calcule la taille moyenne des images, càd la taille 
		    moyenne des images sans les bords noirs """
		
		# Choix du jeu
		if sset == "train":
			target_set = self.trainImages
			set_length = self.trainSetLength
		else:
			target_set = self.testImages
			set_length = self.testSetLength
			
		# Mesures
		average_width = 0.0
		average_height = 0.0
			
		# Pour chaque images
		for i in range(0,set_length):
			height, width = self.getRealImageSize(index = i)
			average_width += width
			average_height += height
			
		# Taille 
		height = round(average_height / set_length)
		width = round(average_width / set_length)
			
		# Retourne le max
		if height >= width:
			return height
		else:
			return width
			
	##############################################################
	# Retourne les séries d'apprentissage
	##############################################################
	def getTrainingSet(self, start = 0, length = 60000):
		""" On retourn le jeu d'entraînement sous forme de séries
		    temporelles """
		
		return self.trainTSImages[start * self.entrySize : start * self.entrySize + length * self.entrySize, : ], self.trainTSLabels[start * self.entrySize : start * self.entrySize + length * self.entrySize, : ]
		
	##############################################################
	# Retourne les séries de tests
	##############################################################
	def getTestSet(self, start = 0, length = 5000):
		""" On retourne le jeux de tests des digits sous forme de 
		    séries temporelles """
		
		return self.testTSImages[start * self.entrySize : start * self.entrySize + length * self.entrySize, : ], self.testTSLabels[start * self.entrySize : start * self.entrySize + length * self.entrySize, : ]
		
	##############################################################
	# Redimmensionne une image et l'affiche
	##############################################################
	def showResizedImage(self, index = 0, size = 20):
		""" On affiche une image dont on change la taille """
		
		im = self.trainImages[index,:,:] * 255
		plt.imshow(resize(im,(size,size)), cmap = cm.Greys_r)
		plt.show()
		
	##############################################################
	# Affiche des informations sur le set
	##############################################################
	def showInfos(self):
		print "Longueur du jeu"
		
	##############################################################
	# Renvoi les bordures de l'image
	##############################################################
	def getImageBorder(self, index = 0, sset = "train"):
		""" On calcule les bords de l'images, càd la position des 
		    premiers pixels non noirs depuis chaque bords """
		
		# Image
		if sset == "train":
			im = self.trainImages[index,:,:]
		else:
			im = self.testImages[index,:,:]
		
		# Resultats
		height_start = 0
		height_end = 0
		width_start = 0
		width_end = 0
		
		# Condition d'arret
		ok = False
		
		# Depuis le haut
		y = 0
		while ok != True:
			if np.extract(im[y,:]>0,im).size > 0:
				height_start = y
				ok = True
			y += 1
			if y >= self.imagesSize:
				ok = True
				
		# Depuis le bas
		ok = False
		y = self.imagesSize - 1
		while ok != True:
			if np.extract(im[y,:]>0,im).size > 0:
				height_end = y
				ok = True
			y -= 1
			if y <= -1:
				ok = True
				
		# Depuis la gauche
		ok = False
		x = 0
		while ok != True:
			if np.extract(im[:,x]>0,im).size > 0:
				width_start = x
				ok = True
			x += 1
			if x >= self.imagesSize:
				ok = True
				
		# Depuis la droite
		ok = False
		x = self.imagesSize - 1
		while ok != True:
			if np.extract(im[:,x]>0,im).size > 0:
				width_end = x
				ok = True
			x -= 1
			if x <= -1:
				ok = True
				
		return height_start, height_end, width_start, width_end
		
	##############################################################
	# Redimmensionne les images
	##############################################################
	def resizeImages(self, size = 0):
		""" On redimenssionne les images chargées """
		
		# Taille
		if size == 0:
			size = self.getAverageImageSize()
		
		# Nouveaux tableau
		newTrainImages = np.zeros((self.trainSetLength,size,size))
		newTestImages = np.zeros((self.testSetLength,size,size))
		
		# Pour chaque image
		for i in range(self.trainSetLength):
			# Bordures
			hs,he,ws,we = self.getImageBorder(i,sset="train")
			hs -= 1
			he += 2
			ws -= 1
			we += 2
			if hs < 0:
				hs = 0
			if he >= self.imagesSize:
				he = self.imagesSize-1
			if ws < 0:
				ws = 0
			if we >= self.imagesSize:
				we = self.imagesSize-1
			
			# Redimensionne
			newTrainImages[i,:,:] = resize(self.trainImages[i,hs:he,ws:we],(size,size))
			
		# Pour chaque image
		for i in range(self.testSetLength):
			# Bordures
			hs,he,ws,we = self.getImageBorder(i,sset="test")
			hs -= 1
			he += 2
			ws -= 1
			we += 2
			if hs < 0:
				hs = 0
			if he >= self.imagesSize:
				he = self.imagesSize-1
			if ws < 0:
				ws = 0
			if we >= self.imagesSize:
				we = self.imagesSize-1
			
			# Redimensionne
			newTestImages[i,:,:] = resize(self.testImages[i,hs:he,ws:we],(size,size))
			
		# Copie
		self.trainImages = newTrainImages.copy()
		self.testImages = newTestImages.copy()
		
		# Nouvelle taille
		self.imagesSize = size
		self.entrySize = size + self.interImagesSpace
		
	##############################################################
	# Tourne les images
	##############################################################
	def rotateImages(self, angle = 90):
		# Nouveaux tableau
		newTrainImages = np.zeros((self.trainSetLength,self.imagesSize,self.imagesSize))
		newTestImages = np.zeros((self.testSetLength,self.imagesSize,self.imagesSize))
		
		# Pour chaque image
		for i in range(self.trainSetLength):
			# Tourne
			newTrainImages[i,:,:] = rotate(self.trainImages[i,:,:],angle)
			
		# Pour chaque image
		for i in range(self.testSetLength):
			# Tourne
			newTestImages[i,:,:] = rotate(self.testImages[i,:,:],angle)
			
		# Copie
		self.trainImages = newTrainImages.copy()
		self.testImages = newTestImages.copy()
		
	##############################################################
	# On arrondi les valeurs des pixels
	##############################################################
	def roundImages(self, threshold = 0.5):
		
		# Pour chaque images d'entraînement
		for i in range(self.trainSetLength):
			for x in range(self.trainImages[i,:,:].shape[0]):
				for y in range(self.trainImages[i,:,:].shape[1]):
					if self.trainImages[i,x,y] >= threshold:
						self.trainImages[i,x,y] = 1.0
					else:
						self.trainImages[i,x,y] = 0.0
						
		# Pour chaque images de test
		for i in range(self.testSetLength):
			for x in range(self.testImages[i,:,:].shape[0]):
				for y in range(self.testImages[i,:,:].shape[1]):
					if self.testImages[i,x,y] >= threshold:
						self.testImages[i,x,y] = 1.0
					else:
						self.testImages[i,x,y] = 0.0
		
	##############################################################
	# Images intégrales
	##############################################################
	def integralImages(self):
		
		# Pour chaque image
		for i in range(self.trainSetLength):
			# Tourne
			self.trainImages[i,:,:] = integral_image(self.trainImages[i,:,:]) / 100.0
			
		# Pour chaque image
		for i in range(self.testSetLength):
			# Tourne
			self.testImages[i,:,:] = integral_image(self.testImages[i,:,:]) / 100.0
			
	##############################################################
	# Applique une transformation affine à une image
	##############################################################
	def affineTransformImage(self, im, scale = (1,1), rotation = 0, shear = (0,0)):
		
		# Centre de l'image
		centerX = im.shape[1]/2.0
		centerY = im.shape[0]/2.0
		
		# Matrice de transformation
		m_trans = np.matrix([[scale[0], shear[0]],[shear[1], scale[1]]])
		
		# Taille
		rows, cols = im.shape[0], im.shape[1]
		
		# Grille
		src_cols = np.linspace(0, cols, 14)
		src_rows = np.linspace(0, rows, 14)
		src_rows, src_cols = np.meshgrid(src_rows, src_cols)
		src = np.dstack([src_cols.flat, src_rows.flat])[0]
		
		# Apply transformation matrix
		dst = src.copy()
		for i in range(0,src.shape[0]):
			src_pos = src[i,:] - np.array([centerX,centerY])
			trans = np.array(m_trans.dot(src_pos).T)
			trans.shape = 2
			trans += np.array([centerX,centerY])
			dst[i,:] = trans
		
		# Piecewise affine transform
		tform = PiecewiseAffineTransform()
		tform.estimate(dst, src)
		
		# Output image
		out_rows = im.shape[0]
		out_cols = im.shape[1]
		out = warp(im, tform, output_shape=(out_rows, out_cols))
		
		# Tourne l'image
		out = rotate(out, angle = rotation)

		return out
		
	##############################################################
	# Applique une transformation élastique à une image
	##############################################################
	def elasticTransformImage(self, im, scale = 6.0, sigma = 1.1):
		
		# Taille
		rows, cols = im.shape[0], im.shape[1]
		
		# Grille
		src_cols = np.linspace(0, cols, 14)
		src_rows = np.linspace(0, rows, 14)
		src_rows, src_cols = np.meshgrid(src_rows, src_cols)
		src = np.dstack([src_cols.flat, src_rows.flat])[0]
		
		# Random deplacement fields
		dep_x = (np.random.rand(14,14)*2.0)-1.0
		dep_y = (np.random.rand(14,14)*2.0)-1.0
		
		# Convolution avec un noyau gaussien
		conv_x = sci.ndimage.filters.gaussian_filter(dep_x, sigma = sigma)
		conv_y = sci.ndimage.filters.gaussian_filter(dep_y, sigma = sigma)
		
		# Multiplication par un facteur d'échelle
		conv_x *= scale
		conv_y *= scale
		
		# Application
		dst = src.copy()
		for x in np.arange(0,14):
			for y in np.arange(0,14):
				dst[x*14+y,0] += conv_x[x,y]
				dst[x*14+y,1] += conv_y[x,y]
		
		# Piecewise affine transform
		tform = PiecewiseAffineTransform()
		tform.estimate(dst, src)
		
		# Output image
		out_rows = im.shape[0]
		out_cols = im.shape[1]
		out = warp(im, tform, output_shape=(out_rows, out_cols))

		return out
		
	##############################################################
	# Applique un filtre gaussien à une image
	##############################################################
	def blurImage(self, im, sigma = 1):
		
		# Copie de l'image
		blurred = im.copy()
		
		# Floute l'image
		blurred = sci.ndimage.filters.gaussian_filter(blurred, sigma = sigma)
		
		return blurred
		
	##############################################################
	# Appliqe un bruit gaussien à une image
	##############################################################
	def noiseImage(self, im, sigma = 0.05):
		
		# Copie de l'image
		noisy = im.copy()
		
		# Taille
		width = noisy.shape[1]
		height = noisy.shape[0]
		
		# Ajoute du bruit
		noisy += np.random.normal(loc = 0.0, scale = sigma, size = (height, width))
		
		return noisy
		
	##############################################################
	# Fusionne deux séries temporelles
	##############################################################
	def mergeTimeseries(self, TS1=0, TS2=1):
		""" On fusionne deux séries en en faisant la moyenne et on 
		    met le résultat dans la première """
		    
		self.trainTSImages[:,TS1*self.entrySize:TS1*self.entrySize+self.entrySize] = (self.trainTSImages[:,TS1*self.entrySize:TS1*self.entrySize+self.entrySize]+self.trainTSImages[:,TS2*self.entrySize:TS2*self.entrySize+self.entrySize])/2.0
		self.testTSImages[:,TS1*self.entrySize:TS1*self.entrySize+self.entrySize] = (self.testTSImages[:,TS1*self.entrySize:TS1*self.entrySize+self.entrySize]+self.testTSImages[:,TS2*self.entrySize:TS2*self.entrySize+self.entrySize])/2.0
		
	##############################################################
	# Duplique une série temporelle
	##############################################################
	def duplicateTimeserie(self, TS=0):
		""" On va dupliquer une série temporelle et la mettre à la 
		    suite """
		    
		self.trainTSImages = np.hstack((self.trainTSImages, self.trainTSImages[:,TS*self.entrySize:TS*self.entrySize+self.entrySize]))
		self.testTSImages = np.hstack((self.testTSImages, self.testTSImages[:,TS*self.entrySize:TS*self.entrySize+self.entrySize]))
		
		self.nbTS += 1
		self.nbInputs = self.nbTS * self.imagesSize
		
	##############################################################
	# Supprime une série temporelle
	##############################################################
	def removeTimeserie(self, TS=0):
		""" On va retirer une série temporelle du jeu de données """
		
		indexes = np.arange(0, TS*self.entrySize)
		indexes = np.append(indexes, np.arange(TS*self.entrySize+self.entrySize, self.trainTSImages.shape[1]))
		self.trainTSImages = self.trainTSImages[:,indexes]
		self.testTSImages = self.testTSImages[:,indexes]
		self.nbTS -= 1
		self.nbInputs = self.nbTS * self.imagesSize
		
	##############################################################
	# Calcule le pourcentage de digits non reconnus
	##############################################################
	def digitErrorRate(self, test, with_miss_array = False, per_digit = False, pos_table = False):
		""" On compare une suite de digits reconnus à ceux de tests
		    et on calcul le digit error rate """
		    
		# Digit count
		n_digit = float(test.size)
		#print "n_digit : {}".format(n_digit)
		# Miss count
		miss_count = 0.0
		
		# Tableau de digit ratés
		miss_array = np.zeros((test.size,2))
		
		# Position des digits
		pos_array = []
		
		# Raté par digit
		digits_count = np.zeros(10, dtype='float64')
		digits_miss = np.zeros(10, dtype='float64')

		# For each digit
		for i in np.arange(0, n_digit):
			digits_count[self.testLabels[i]] += 1.0
			miss_array[i,0] = test[i]
			miss_array[i,1] = self.testLabels[i]
			# Test
			#print "{} ?? {}".format(test[i],self.testLabels[i])
			if int(test[i]) != int(self.testLabels[i]):
				miss_count += 1.0
				#miss_array = np.append(miss_array, (test[i],self.testLabels[i]))
				digits_miss[self.testLabels[i]] += 1.0
				pos_array = np.append(pos_array, int(i))
		#print "miss_count : {}".format(miss_count)
		# Raté par digit
		miss_per_digit = (digits_miss / digits_count) * 100.0
		
		# Error rate
		if with_miss_array == True and per_digit == True and pos_table == True:
			return (float(miss_count/n_digit)*100.0, miss_array, miss_per_digit, pos_array)
		elif with_miss_array == True and per_digit == True:
			return (float(miss_count/n_digit)*100.0, miss_array, miss_per_digit)
		elif with_miss_array == True and pos_table == True:
			return (float(miss_count/n_digit)*100.0, miss_array, pos_array)
		elif per_digit == True and pos_table == True:
			return (float(miss_count/n_digit)*100.0, miss_per_digit, pos_array)
		elif with_miss_array:
			return (float(miss_count/n_digit)*100.0, miss_array)
		elif per_digit == True:
			return (float(miss_count/n_digit)*100.0, miss_per_digit)
		else:
			return float(miss_count/n_digit)*100.0
			
	##############################################################
	# Calcule le pourcentage de digits non reconnus sur le jeu
	# d'entraînement
	##############################################################
	def digitErrorRateTrainging(self, train, with_miss_array = False, per_digit = False, pos_table = False):
		""" On compare une suite de digits reconnus à ceux d'entrainements
		    et on calcul le digit error rate """
		    
		# Digit count
		n_digit = float(train.size)

		# Miss count
		miss_count = 0.0
		
		# Tableau de digit ratés
		miss_array = []
		
		# Position des digits
		pos_array = []
		
		# Raté par digit
		digits_count = np.zeros(10, dtype='float64')
		digits_miss = np.zeros(10, dtype='float64')

		# For each digit
		for i in np.arange(0, n_digit):
			digits_count[self.trainLabels[i]] += 1.0
			# Test
			if int(train[i]) != int(self.trainLabels[i]):
				miss_count += 1.0
				miss_array = np.append(miss_array, (train[i],self.trainLabels[i]))
				digits_miss[self.trainLabels[i]] += 1.0
				pos_array = np.append(pos_array, int(i))

		# Raté par digit
		miss_per_digit = (digits_miss / digits_count) * 100.0
		
		# Error rate
		if with_miss_array == True and per_digit == True and pos_table == True:
			return (float(miss_count/n_digit)*100.0, miss_array, miss_per_digit, pos_array)
		elif with_miss_array == True and per_digit == True:
			return (float(miss_count/n_digit)*100.0, miss_array, miss_per_digit)
		elif with_miss_array == True and pos_table == True:
			return (float(miss_count/n_digit)*100.0, miss_array, pos_array)
		elif per_digit == True and pos_table == True:
			return (float(miss_count/n_digit)*100.0, miss_per_digit, pos_array)
		elif with_miss_array:
			return (float(miss_count/n_digit)*100.0, miss_array)
		elif per_digit == True:
			return (float(miss_count/n_digit)*100.0, miss_per_digit)
		else:
			return float(miss_count/n_digit)*100.0
			
	##############################################################
	# Calcule le pourcentage de digits non reconnus
	##############################################################
	def digitErrorRateBis(self, input_signal, target_signal):
		""" On compare une suite de digits reconnus à ceux de tests
		    et on calcul le digit error rate """
		    
		# Digit count
		n_digit = float(input_signal.size)

		# Miss count
		miss_count = 0.0

		# For each digit
		for i in np.arange(0, n_digit):
			# Test
			if int(input_signal[i]) != int(self.testLabels[i]):
				miss_count += 1.0
				
		# Error rate
		return float(miss_count/n_digit)*100.0
		
