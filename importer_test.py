#!/usr/bin/env python
# -*- coding: utf-8 -*-

##########################################################################
#
# Auteur : Nils Schaetti <nils.schaetti@univ-comte.fr>
# Date : 28.04.2015 20:22:49
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
import math

#########################################################################
#
# Ici on teste l'importation et la traitement des chiffres MNIST
#
#########################################################################

####################################################
# Fonction principale
####################################################
if __name__ == "__main__":
	
	# Charge
	print "Chargement depuis {}...".format(sys.argv[1])
	digitImport = MNISTImporter()
	digitImport.Load(sys.argv[1])
	
	# Double la taille du set avec 
	# des transformations affines
	for i in np.arange(0,1):
		print "################################# Génération de la {} ème extension ###########################".format(i)
		digitImport.addAffineImages(verbose = 1)
	
	# Affiche
	for i in np.arange(60000,120000,1000):
		digitImport.showImage(index = i)
	
	# Sauve dans un fichier
	print "Enregistrement dans {}...".format(sys.argv[2])
	digitImport.Save(sys.argv[2])

