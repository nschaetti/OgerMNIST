#!/usr/bin/env python
# -*- coding: utf-8 -*-

##########################################################################
#
# Auteur : Nils Schaetti <nils.schaetti@univ-comte.fr>
# Date : 14.05.2015 18:20:23
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
from PIL import Image, ImageTk
import Tkinter as Tk

############################################################################
# Un outil pour modifier le jeu MNSIT
############################################################################

im_pos = 0

# Vérifie les params
if len(sys.argv) != 2:
	print "Donnez un nom de fichier!"
	exit()

# Import les digits
print "Importation des digits depuis {}".format(sys.argv[1])
digitImport = MNISTImporter()
digitImport.Load(sys.argv[1])

root = Tk.Tk()

root.title("MNISTool")

#image = Image.open("lena.jpg")
image = Image.fromarray(digitImport.trainImages[im_pos,:,:]*255)
photo = ImageTk.PhotoImage(image)

canvas = Tk.Canvas(width=512,height=512)
canvas.create_image(256,256, image = photo)
canvas.grid(row = 0, column = 0, rowspan = 6)

def callback_next(event):
	global im_pos
	global v
	global digitImport
	global image
	global photo
	global canvas
	im_pos += 1
	v.set(str(digitImport.trainLabels[im_pos]))
	image = Image.fromarray(digitImport.trainImages[im_pos,:,:]*255)
	photo = ImageTk.PhotoImage(image)
	canvas.create_image(256,256, image = photo)

but_prev = Tk.Button(text="Prev")
but_prev.grid(row = 0, column = 1)

v = Tk.StringVar()
label = Tk.Label(textvariable=v)
v.set(str(digitImport.trainLabels[im_pos]))
label.grid(row = 0, column = 2)

n = 1
for i in range(1,5):
	for j in range(1,5):
		Tk.Button(text=str(n)).grid(row = i, column = j)
		n += 1
		
but_next = Tk.Button(text="Next")
but_next.grid(row = 5, column = 1)
but_next.bind("<Button-1>", callback_next)

root.mainloop()
