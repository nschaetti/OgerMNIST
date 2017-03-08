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
	
	# Affiche infos
	#digitImport.showTrainingSet(start=0, length=20)
	
	digitImport.resizeImages(size = 18)
	digitImport.rotateImages(angle = 30)
	digitImport.resizeImages(size = 18)
	digitImport.rotateImages(angle = 60)
	
	digitImport.showImage(index = 0)
	digitImport.showImage(index = 1)
	digitImport.showImage(index = 2)
	
	

