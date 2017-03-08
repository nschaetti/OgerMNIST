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
# On crée le fichier contenant les longueurs des données
###########################################################################

# Fonction principale
if __name__ == "__main__":
	
	# Paramètres
	nInputs = int(sys.argv[1])
	nbFiles = int(sys.argv[2])
	fichier = sys.argv[3]
	
	# Nouveau fichier
	mon_fichier = open(fichier,"w")
	
	# Pour chaque fichier
	for i in np.arange(0,nbFiles):
		mon_fichier.write("   " + str(float(nInputs)) + "\n")
	
	# Ferme le fichier
	mon_fichier.close()
