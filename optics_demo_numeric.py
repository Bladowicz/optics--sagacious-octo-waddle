# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:14:11 2015

@author: artur
"""


# NAJLEPIEJ NIE PUSZCZAC CALEGO SKRYPTU PRZEZ F5, TYLKO ZAZNACZAC POSZCZEGOLNE FRAGMENTY 
# I UZYWAC F9

from sklearn.datasets.samples_generator import make_blobs
import pylab as pl
import pandas as p
import optics as op
import numpy as np


# generowanie zbioru obserwacji, skupienia o srodkach w punktach 'centers'
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_tru = make_blobs(n_samples=750, centers=centers, cluster_std=0.4)

# typ odleglosci - dostepne 'euclidean' i 'hamming'
dtype = "euclidean"

name = './list-numeric-eps_'+str(epsilon)+'_mpts_'+str(minpts)+'.txt'
name = './list-numeric-eps_'+str(epsilon)+'_mpts_'+str(minpts)+'.txt', names=['i', 'rd']

minpts = 5
epsilon = 0.7
testtree = op.setOfObjects(X) #  stworzenie obiektu klasy setOfObjects na podstawie obserwacji
op.prep_optics(testtree, epsilon, minpts, dtype) # dla kazdego punktu znajduje otoczenie 
                        # czyli takie punkty, dla ktorych odleglosc od danego punktu jest <= epsilon     
                        # oraz dla kazdego punktu z tego otoczenia oblicza core distance
                        # (tu jest wykorzystywany parametr minpts)
op.build_optics(testtree, epsilon, minpts, name, dtype)
                        # w tej funkcji obliczane sa reachability distance, na podstawie ktorych
                        # ustalany jest odpowiedni porzadek punktow
f = p.read_csv(name, names=['i', 'rd'])
                        # wynik jest zapisywany w powyzszym pliku txt, zawiera on dwie kolumny:
                        # w pierwszej id punktu, w drugiej reachability distance danego punktu
                        # UWAGA! przy kazdym uzyciu trzeba plik usunac, poniewaz zawartosc jest dodawana,
                        # a nie nadpisywana
# wykres (id puntu, reachability distance)
rd = f['rd']
pl.plot(rd)
pl.title('epsilon = ' + str(epsilon) + ', minpts = ' + str(minpts))
pl.show()
        

# zastosowanie algorytmu dla roznych wartosci parametrow minpts i epsilon
#for minpts in [3, 5, 7, 10]:
#    for epsilon in [0.3, 0.5, 0.7, 0.9, 1.2]:
#        testtree = op.setOfObjects(X) #  stworzenie obiektu klasy setOfObjects na podstawie obserwacji
#        op.prep_optics(testtree, epsilon, minpts, dtype) # dla kazdego punktu znajduje otoczenie 
#                                # czyli takie punkty, dla ktorych odleglosc od danego punktu jest <= epsilon     
#                                # oraz dla kazdego punktu z tego otoczenia oblicza core distance
#                                # (tu jest wykorzystywany parametr minpts)
#        op.build_optics(testtree, epsilon, minpts,'./list-numeric-eps_'+str(epsilon)+'_mpts_'+str(minpts)+'.txt', dtype)
#                                # w tej funkcji obliczane sa reachability distance, na podstawie ktorych
#                                # ustalany jest odpowiedni porzadek punktow
#        f = p.read_csv('./list-numeric-eps_'+str(epsilon)+'_mpts_'+str(minpts)+'.txt', names=['i', 'rd'])
#                                # wynik jest zapisywany w powyzszym pliku txt, zawiera on dwie kolumny:
#                                # w pierwszej id punktu, w drugiej reachability distance danego punktu
#                                # UWAGA! przy kazdym uzyciu trzeba plik usunac, poniewaz zawartosc jest dodawana,
#                                # a nie nadpisywana
#        # wykres (id puntu, reachability distance)
#        rd = f['rd']
#        pl.plot(rd)
#        pl.title('epsilon = ' + str(epsilon) + ', minpts = ' + str(minpts))
#        pl.show()
        
      


# na podstawie powyzszego wykresu nalezy wybrac odpowiedni epsilon_prime, tj. taki dla ktorego
# otrzymujemy rozsadny podzial na skupienia    

eps_prime = 0.14 # u mnie tyle bylo spoko
op.ExtractDBSCAN(testtree, epsilon_prime=eps_prime)
# ta funkcja przypisuje kazdemu punktowi skupienie, do ktorego nalezy na podstawie ustalonego epsilona
# id skupienia -1 jest przypisywane punktom uznanym za szum
print "Przypisane skupienia:"
print testtree._cluster_id

print sorted(list(set(testtree._cluster_id)))




# Rysowanie wykresu obserwacji podzielonych na skupienia

# Core samples and labels #
core_samples = testtree._index[testtree._is_core[:] > 0]
labels = testtree._cluster_id[:]
#len(testtree._index[testtree._is_core[:] > 0])
n_clusters_ = max(testtree._cluster_id) # gives number of clusters
n_clusters_

n_clusters_ = max(testtree._cluster_id) # gives number of clusters
n_clusters_

# Plot results #
pl.figure()

# Black removed and is used for noise instead.
unique_labels = set(testtree._cluster_id[:]) # modifed from orginal #
colors = pl.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'
        markersize = 6
    class_members = [index[0] for index in np.argwhere(labels == k)]
    cluster_core_samples = [index for index in core_samples
                            if labels[index] == k]
    for index in class_members:
        x = X[index]
        if index in core_samples and k != -1:
            markersize = 14
        else:
            markersize = 6
        pl.plot(x[0], x[1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=markersize)

pl.title('Estimated number of clusters for eps_prime = %f: %d' % (eps_prime, n_clusters_))
pl.show()
