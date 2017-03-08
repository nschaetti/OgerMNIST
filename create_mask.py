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
# On crée un masque d'entrée pour le NTC (Nonlinear Transient Computing)
###########################################################################

# PARAMTERS
random_values = np.array([0.1, -0.1])

# Fonction principale
if __name__ == "__main__":
	
	# Paramètres
	sparsity = float(sys.argv[1])
	nNode = int(sys.argv[2])
	nInputs = int(sys.argv[3])
	fichier = sys.argv[4]
	
	# Nouveau tableau
	t_mat = np.zeros((nNode,nInputs))
	
	# For each node
	for n in np.arange(0,nNode):
		for i in np.arange(0,nInputs):
			if np.random.rand() < sparsity:
				t_mat[n,i] = random_values[np.random.random_integers(0,random_values.size-1)]
	
	# Matrice
	mat = np.matrix(t_mat)
	
	# Sauve
	io.savemat(fichier, mdict={'Wi' : mat})
