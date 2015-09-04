#!/usr/bin/python
import optics
from sklearn.datasets.samples_generator import make_blobs
import pylab as pl
import pandas as p
import optics as op
import numpy as np
import os
def get_distance(in_file):
    fh = open(in_file)
    labels = map(lambda x: x.strip('"'), fh.readline().strip().split(','))
    z = len(labels)
    out = {}
    for nr, l in enumerate(fh):
        l = l.strip().split(',')
        for x in range(nr+1, z):
            out[tuple(sorted((labels[x],labels[nr])))] = float(l[x])
    return (labels, out)



if __name__=="__main__":
    minpts = 5
    #epsilon = 0.7
    #epsilon = 1.5
    epsilon = 5
    outfile = 'test.txt'
    if os.access(outfile, 0): os.remove(outfile)

    labels, distance = get_distance( 'lifts.csv')

    testtree = optics.setOfObjects(labels, distance)
    testtree.prep_optics(epsilon, minpts)
    testtree.build_optics(epsilon, minpts, outfile)




    f = p.read_csv(outfile, names=['i', 'rd'])
    rd = f['rd']
    pl.plot(rd)
    pl.title('epsilon = ' + str(epsilon) + ', minpts = ' + str(minpts))
    pl.show()
