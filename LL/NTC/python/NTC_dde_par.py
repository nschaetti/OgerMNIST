#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pylab
import os
import cPickle
import struct
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import scipy.io as sio
import numpy as np
import math as math
import ctypes
from scipy.interpolate import interp1d

# Simulation des données d'entrée de la dynamique à retard passe-bas simple
# délai, et de la réponse dynamique à cette entrée
# Les digits sont traités avec un ordre aléatoire qui utilise la
# permutation effectuée de toute façon pour le choix aléatoire des
# partitions de test et d'apprentissage.

################################################
# PARAMETERS SECTION
################################################

Nttmax = 20								# How many groups of spoken digits
Ndptt = 25								# How many spoken digits in a group
Nnode = 306								# How many nodes (neurons)
spsty = 0.1								# Sparsity of inputs to reservoir
Nsdp = 160								# Single spoken digit length
Cmax = 100								# Research step for Vp parameter
Vpmin = -0.00025						# Minimum value for Vp parameter
Vpmax = 0.00025							# Maximum value for Vp parameter
rho = 2.5								# Ratio tau/delta
h = 0.01								# Integration timestep (in units of tau)
Nhpn = math.floor(1.0/h/rho)			# Timesteps between two nodes
nbe = Nnode * Nhpn						# Timesteps per delay
tau = 1.0/h								# Internal response of the dynamics
nl = 0									# ?
beta = 1.1								# See equation 2
phi = 0.01								# See equation 2
alpha = 1								# See equation 2
Delta = 2.5								#
Feedback_on = 1							# Feedback activated
N0 = 0									# Time origin for cutting each digit in the sequence
dT0 = 36								# Time shift for choosing node position between several possible ones, with
										# a constant spacing between the nodes

################################################
# FUNCTIONS PART
################################################

#
# Load .dat files
#
def loadLengthFile(dat_file):
	# Open file
	fd = open(dat_file,"r")
	
	# Read lines
	content = fd.readlines()
	
	# Empty data
	data = np.zeros(len(content), dtype='float64')
	
	# For each line
	pos = 0
	for i in content:
		data[pos] = float(i.strip())
		pos += 1
	
	# Close file
	fd.close()
	
	return data

#
# Load Cochleagram file
#
def loadCochleagreamFile(coch_file, coch_length):
	# Open file
	fd = open(coch_file,"r")
	
	# Read lines
	lines = fd.readlines()
	
	# 86 x coch_length array
	data = np.zeros((86, coch_length), dtype='float64')
	
	# For each 86 fourier channels
	for channel in np.arange(0,86):
		inputs = lines[channel].split("   ")
		# For each inputs
		for i in np.arange(0,int(coch_length)):
			data[channel,i] = float(inputs[i+1])
	
	# Close file
	fd.close()
	
	return np.matrix(data)
	
#
# Function
#
def zero_func(x):
	return x - beta * (math.sin(x + phi))^2

################################################
# MAIN PART
################################################
if __name__ == "__main__":
	
	# Load lengths of each 500 spoken digits
	print "Loading spoken digits length file..."
	digtlen = loadLengthFile("./inputs/load_lengthfile1.dat")
	Nsd = digtlen.size
	
	# Value of the Vp parameter to test
	Vp = np.linspace(Vpmin, Vpmax, Cmax)
	ResVp = np.zeros((Cmax,3))
	
	# Load matrice file containing input connectivity mask
	print "Loading input connectivity mask..."
	Wi = sio.loadmat("./inputs/demo3/mask_" + str(Nnode) + ".mat")['Wi']
	
	# Constructing the input with the mask
	nMax = 0
	nMin = 0
	
	# For each cochleagram file
	for m in np.arange(1,Nsd+1):
		
		# Load a chochleagram
		Mc = loadCochleagreamFile("./inputs/load_inputs1_" + str(m) + ".dat", digtlen[m-1])
		
		# Construct input with mask
		Mu = Wi * Mc
		
		# Max / min
		nMax = max(Mu.max(), nMax)
		nMin = min(Mu.min(), nMin)
		
	# Normalization factor
	nWi = nMax - nMin
	print "Normalization factor for the input : " + str(nWi)
	Wi = Wi / nWi
	
	# Random permutation of all digits
	prm = np.random.permutation(Nsd) + 1
	
	# Scan varied parameter Vp
	for C in np.arange(0,Cmax):
		
		# Open or closed loop operation
		if Feedback_on == 1:
			print "Closed loop operation (with delayed feedback"
		else:
			print "Open loop operation"
			
		# Small change in the pitch
		Npitch = Vp[C]
		
		# Concatenation matrix of the transcients of the successive digits
		A = []
		B = []
		
		# Parameters for Runge-Kutta
		tps = [tau, nbe, 0, Nspd * Ndptt]
		param = [nl, beta, phi, alpha, Feedback_on]
		
		# Determine the stable steady state, to avoid initial "unrealistic"
		# transcient for the first digit of the sequence
		cdi = sio.fmin_ncg(zero_func, 0.5)
		
		# Counters for the digit answers vectors (train and learn)
		Dgv2 = np.zeros((Nttmax,Ndptt))
		
		# Loop for the Nttmax sequences of each Ndptt digits to be processed
		for p in np.arange(1,Nttmax+1):
			
			# Vector for an input sequence of Ndptt digits to be processed
			ttrace = np.zeros(Nspd * nbe * Ndptt)
			
			# Loop for Ndptt digit in a single sequence
			for m in np.arange(1,Ndptt+1):
				
				# Load a cochleagram
				index = prm[Ndptt * (p-1) + m])
				print "Loading cochleagram file ./inputs/load_inputs1_" + str(index) + ".dat"
				Mc = loadCochleagreamFile("./inputs/load_inputs1_" + str(index) + ".dat", digtlen[index-1])
				
				# Construct input with mask
				Mu = Wi * Mc
				
				# For each input signal
				for k in np.arange(0, Mu.shape[1]):
					# For each node
					for l in np.arange(0,Nnode):
						ttrace[Nspd * nbe * (m-1) * k * nbe + Nhpn * l : Nspd * nbe * (m-1) * k * nbe + Nhpn * (l+1)] = Mu[l,k] * np.ones(Nhpn)
						
			# Load shared library
			rk4_lib = ctypes.CDLL("./rk4_ntc.mexa64")
			
			# Calculate the transient nonlinear delay dynamics for one sequence
			yc = rk4_ntc(cdi * np.ones(nbe+1), tps, param, Delta * ttrace)
			
			# Loop for extracting the 2D pattern from the ReadOut transcient,
			# for each of the Ndptt digits in the sequence
			DD = np.zeros((1,Ndptt))
			for m in np.arange(1,Ndptt+1):
				
				# Define the position of the digit in the ordered data base
				no = prm[Ndptt * (p-1) + m]
				
				# Store the digit value of currently processed response
				dgo = no - math.floor((no-1.0)/10.0)*10.0
				DD[m] = dgo
				
				# Recover the digit length of the currently processed response
				Kpd = digtlen[no]
				
				# Starting date of the current digit in the sequence
				start = (m-1) * nbe * Nspd + 1 + N0 + dT0
				
				# Sample dates for the currently processed digit + 1 delay
				tnop = np.arange(start+1,start+((Kpd+1)*Nnode*Nhpn)+1)
				yci = interp1
