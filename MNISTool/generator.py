#!/usr/bin/env python
# -*- coding: utf-8 -*-

##########################################################################
#
# Auteur : Nils Schaetti <nils.schaetti@edu.univ-comte.fr>
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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import math
from scipy import io
import numpy as np

###########################################################################
# Crée les fichiers de digits et le fichier de correspondance
###########################################################################

# Enregistre une image
def saveImage(target_file, image):
	print "Saving image " + target_file
	
	# Nouveau fichier
	mon_fichier = open(target_file,"w")
	
	# Pour chaque lignes
	for j in np.arange(image.shape[1]):
		# Pour chaque colonnes
		for i in np.arange(image.shape[0]):
			mon_fichier.write("   " + str(image[i,j]))
		mon_fichier.write("\n")
	
	# Ferme le fichier
	mon_fichier.close()
	
# Enregistre le fichier de correspondances
def saveLabels(target_file, labels):
	print "Saving corr file " + target_file
	
	# Nouveau fichier
	mon_fichier = open(target_file,"w")
	
	# Pour chaque labels
	for i in np.arange(0, labels.shape[0]):
		mon_fichier.write("   " + str(labels[i]) + "\n")
	
	# Ferme le fichier
	mon_fichier.close()

# Enregistre la description
def saveDesc(target_dir, image_size, rep):
	
	# Nouveau fichier 
	mon_fichier = open(target_dir+"/desc","w")
	
	# Ecrits
	mon_fichier.write(str(image_size) + "\n")
	mon_fichier.write(str(rep))
	
	# Ferme le fichier
	mon_fichier.close()

# Fonction principale
if __name__ == "__main__":
	
	# Paramètres
	mnist_file = sys.argv[1]
	target_dir = sys.argv[2]
	image_size = int(sys.argv[3])
	nb_train = int(sys.argv[4])
	nb_test = int(sys.argv[5])
	
	# Charge les digits
	digitImport = MNISTImporter()
	digitImport.Load(sys.argv[1])
	
	# Séries
	digitImport.resizeImages(size = image_size)
	digitImport.addTimeserie()
	digitImport.rotateImages(angle = 30)
	digitImport.addTimeserie()
	digitImport.rotateImages(angle = 60)
	digitImport.addTimeserie()
	digitImport.rotateImages(angle = 60)
	digitImport.addTimeserie()
	digitImport.generateLabels()
	
	# Crée les répertoires
	os.makedirs(target_dir + "/images")
	os.makedirs(target_dir + "/test")
	
	# Enregistre la description
	print "Saving description..."
	saveDesc(target_dir, image_size, 4)
	
	# Pour chaque image
	print "Saving training images..."
	for i in np.arange(0, nb_train):
		saveImage(target_dir + "/images/image_" + str(i+1) + ".dat", digitImport.trainTSImages[i*image_size:i*image_size+image_size,:])
	print "Saving test images..."
	for i in np.arange(0, nb_test):
		saveImage(target_dir + "/test/image_" + str(i+1) + ".dat", digitImport.testTSImages[i*image_size:i*image_size+image_size,:])
		
	# Correspondances
	print "Saving labels..."
	saveLabels(target_dir + "/train_labels", digitImport.trainLabels[0:nb_train])
	saveLabels(target_dir + "/test_labels", digitImport.testLabels[0:nb_test])

