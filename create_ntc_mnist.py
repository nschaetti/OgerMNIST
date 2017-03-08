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
	print image.shape
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
		mon_fichier.write("   " + str(labels[i]+1) + "\n")
	
	# Ferme le fichier
	mon_fichier.close()

# Fonction principale
if __name__ == "__main__":
	
	# Paramètres
	mnist_file = sys.argv[1]
	target_dir = sys.argv[2]
	image_size = int(sys.argv[3])
	nb_digits = int(sys.argv[4])
	corr_file = sys.argv[5]
	
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
	digitImport.showTrainingSet(length = 10)
	
	# Pour chaque image
	for i in np.arange(0, nb_digits):
		saveImage(target_dir + "/load_inputs1_" + str(i+1) + ".dat", digitImport.trainTSImages[i*image_size:i*image_size+image_size,:])
		
	# Correspondances
	saveLabels(corr_file, digitImport.trainLabels[0:nb_digits])
