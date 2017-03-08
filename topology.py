#!/usr/bin/env python
# -*- coding: utf-8 -*-

##########################################################################
#
# Auteur : Nils Schaetti <nils.schaetti@univ-comte.fr>
# Date : 11.06.2015 23:03:32
# Lieu : Nyon, Suisse
# 
# Fichier sous licence GNU GPL
#
###########################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm

#
# Crée une matrix d'un réseau small-world avec le modèle de Watts et Strogatz
#
def createSmallWorldNetwork(size = 100, degree = 10, beta = 0.5):
	
	# Matrice vide
	w = np.zeros((size,size))
	
	# Crée un anneau de taille "size" avec K voisins
	for n in np.arange(0,size):
		for v in np.arange(1,degree/2+1):
			
			# Position des voisins
			down = (n - v) % size
			
			# Index
			w[n,down] = np.random.rand()
	
	# Rebranche chaque noeud
	for i in np.arange(0,size):
		for j in np.arange(i-(degree/2),i):
			
			target = j % size
			
			if w[i,target] > 0:
				# Prob
				if np.random.rand() <= beta:
					while True:
						k = np.random.randint(0,size)
						if k != i:
							break
					w[i,k] = w[i,target]
					w[i,target] = 0
	
	#plt.imshow(w, cmap = cm.Greys_r)
	#plt.show()
	
	return w

#
# Retourne le degré d'un noeud
#
def getNodeDegree(w, n = 0):
	count = 0
	for i in np.arange(0,w.shape[0]):
		if w[i,n] > 0:
			count += 1
	for i in np.arange(0,w.shape[0]):
		if w[n,i] > 0:
			count += 1
	return count

#
#
# Retourne le total des degrés
def getTotalDegree(w):
	count = 0
	for i in np.arange(0,w.shape[0]):
		count += getNodeDegree(w, i)
	return count

#
# Crée une matrix d'un réseau scale-free avec un degré suivant une loi de puissance
#
def createScaleFreeNetwork(size = 100, m0 = 10, m = 10):
	
	# Matrice vide
	w = np.zeros((size,size))
	
	# Degrees
	degrees = np.zeros(size)
	total_degree = 0
	
	# Start
	for i in np.arange(0,m0):
		for j in np.arange(0,m0):
			w[i,j] = np.random.rand()
			degrees[i] += 1
			total_degree += 1
	
	# Pour chaque noeud
	for i in np.arange(m0,size):
		count = 0
		pos = 0
		while count < m:
			if w[i,pos] == 0:
				pi = degrees[pos] / total_degree
				if np.random.rand() <= pi:
					w[i,pos] = np.random.rand()
					degrees[i] += 1
					w[pos,i] = np.random.rand()
					degrees[pos] += 1
					total_degree += 2
					#w[i,pos] = 1
					count += 1
			
			# Noeud suivant
			pos += 1
			if pos == i:
				pos = 0
	
	#plt.imshow(w, cmap = cm.Greys_r)
	#plt.show()
	#print degrees
	
	return w
