# -*- coding: utf-8 -*-
import sys
import scipy
import numpy as np
import itertools as it
from math import sqrt
from sklearn.metrics import hamming_loss
from sklearn.neighbors import BallTree
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

## Main Class ##

class setOfObjects(object):

    def __init__(self,data_points, dist_dict):
        self._data_points   =   data_points
        self._n             =   len(data_points)
        self._dist_dict     =   dist_dict
        self._processed     =   scipy.zeros((self._n,1),dtype=bool) ## Start all points as 'unprocessed' ##
        self._reachability  =   scipy.ones(self._n)*scipy.inf       ## Important! ##
        self._core_dist     =   scipy.ones(self._n)*scipy.nan
        self._index         =   scipy.array(range(self._n))         ## Might be faster to use a list? ##
        self._neighbors     =   dict(zip(range(self._n), [[] for i in range(self._n)]))
        self._nneighbors    =   scipy.ones(self._n,dtype=int)
        self._cluster_id    =   -scipy.ones(self._n,dtype=int)      ## Start all points as noise ##
        self._is_core       =   scipy.ones(self._n,dtype=bool)
        self._ordered_list  =   []                                  ### DO NOT switch this to a hash table, ordering is important ###

    def _set_neighborhood(self, point, epsilon, dtype):
        self._neighbors[point] = self.get_neighbors_dist(point, epsilon, dtype)[0]
        self._nneighbors[point] = len(self._neighbors[point])

    def _set_core_dist(self, point, MinPts, dtype):
        dist = []
        for j in self._neighbors[point]:
            dist.append(self.distance(self._data_points[point], self._data_points[j], dtype))
        self._core_dist[point] = min_k(dist, MinPts)

    def prep_optics(self, epsilon, MinPts, dtype = "euclidean"):
        for i in self._index:
            self._set_neighborhood(i, epsilon, dtype)
        for j in self._index:
            if self._nneighbors[j] >= MinPts:
                self._set_core_dist(j, MinPts, dtype)
        print('Core distances and neighborhoods prepped for ' + str(self._n) + ' points.')

    def get_neighbors_dist(self, point, epsilon, dtype):
        neigh = []
        dist = []
        for j in it.ifilter(lambda x: x!=point, self._index):
            d = self.distance(self._data_points[point], self._data_points[j], dtype)
            if d <= epsilon:
                neigh.append(j)
                dist.append(d)
        return neigh, dist

    def distance(self, x, y, dtype):
        try:
            return self._dist_dict[tuple(sorted((x, y)))]
        except KeyError as e:
            for k,v in  self._dist_dict.iteritems():
                print k,v
            print (x, y)
            raise(e)

    def build_optics(self, epsilon, MinPts, Output_file_name, dtype = "euclidean"):
        for point in self._index:
            if self._processed[point] == False:
                self.expandClusterOrder(point, epsilon, MinPts, Output_file_name, dtype)


    def expandClusterOrder(self, point, epsilon, MinPts, Output_file_name, dtype):
        if self._core_dist[point] <= epsilon:
            while not self._processed[point]:
                self._processed[point] = True
                self._ordered_list.append(point)
                with open(Output_file_name, 'a+') as file:
                    file.write((str(point) + ', ' + str(self._reachability[point]) + '\n'))
                    point = self.set_reach_dist(point, epsilon, dtype)
        else:
            self._processed[point] = True    # Probably not needed... #

    def set_reach_dist(self, point_index, epsilon, dtype):
        indices, distances = self.get_neighbors_dist(point_index, epsilon, dtype)
        if scipy.iterable(distances):
            unprocessed = ((self._processed[indices] < 1).T)[0].tolist()
            unprocessed = filter_list(indices, unprocessed)
            rdistances = scipy.maximum(filter_list(distances, unprocessed),self._core_dist[point_index])
            self._reachability[unprocessed] = scipy.minimum(self._reachability[unprocessed], rdistances)
            if len(unprocessed) > 0:
                return sorted(zip(self._reachability[unprocessed],unprocessed), key=lambda reachability: reachability[0])[0][1]
            else:
                return point_index
        else:
            return point_index

    def ExtractDBSCAN(self, epsilon_prime):
        cluster_id = 0
        for entry in self._ordered_list:
            if self._reachability[entry] > epsilon_prime:
                if self._core_dist[entry] <= epsilon_prime:
                    cluster_id += 1
                    self._cluster_id[entry] = cluster_id
                else:
                    self._cluster_id[entry] = -1
            else:
                self._cluster_id[entry] = cluster_id
                if self._core_dist[entry] <= epsilon_prime:
                    self._is_core[entry] = 1
                else:
                    self._is_core[entry] = 0


def min_k(x, k):
    x.sort()
    return x[k-1]


def filter_list(x, y):
    z = []
    for v1, v2 in [ (v1, v2) for v1, v2 in zip(x, y)]:
        if v2:
            z.append(v1)
    return z
