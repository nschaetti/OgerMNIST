import Oger
import pylab
import mdp
import numpy as np

# Reservoir de demonstration pour la tache
# NARMA d'ordre 10.

# Genere le jeu d'entrainement et de test
u_train,y_train = Oger.datasets.narma10(n_samples = 1, sample_len = 5000)
u_test,y_test = Oger.datasets.narma10(n_samples = 1, sample_len = 1000)

# Construction du reservoir et de la couche
# de sortie.
reservoir = Oger.nodes.ReservoirNode(output_dim=100)
readout = Oger.nodes.RidgeRegressionNode()

# Construit le flux
flow = mdp.Flow([reservoir, readout], verbose=1)

# Le jeu d'entrainement pour chaque noeud du reseau
d = zip(u_train,y_train)
print np.array(d).shape
data = [None, zip(u_train,y_train)]

# Entraine le flux
flow.train(data)

# Applique le reseau sur le jeu de test
y_hat = flow(u_test)

# Mesure l'erreur du reseau
print "NRMSE: " + str(Oger.utils.nrmse(y_test[0], y_hat))
print "NMSE: " + str(Oger.utils.nmse(y_test[0], y_hat))
print "RMSE: " + str(Oger.utils.rmse(y_test[0], y_hat))
print "MSE: " + str(Oger.utils.mse(y_test[0], y_hat))
