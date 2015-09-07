#!/usr/bin/python
from src import *
from sklearn.datasets.samples_generator import make_blobs
import pylab as pl
import pandas as p
import optics as op
import numpy as np
import os, itertools
from math import sqrt



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


def get_from_list(in_file):
    with open(in_file) as fh:
        labels = []
        out = {}
        temp = []
        for nr, line in enumerate(fh):
            line = line.strip().split(',')
            temp.append(map(float, line[:4]))
            labels.append(line[4])
        for x,y in itertools.combinations(range(len(labels)), 2):
            out[tuple(sorted((x,y)))] = sqrt(sum([(each[0] - each[1])**2 for each in  zip(temp[x], temp[y])]))


        return (labels, out)

def get_from_temp(in_file):
    with open(in_file) as fw:
        out = {}
        temp = set()
        for line in fw:
            line = line.strip().split()
            out[(int(line[0]), int(line[1]))] = float(line[2])
            temp.add(line[0])
        temp = sorted(map(int, temp))
        return (temp, out)



if __name__=="__main__":
    minpts = 10
    epsilon = 0.7
    #epsilon = 1.5
    #epsilon = 5
    outfile = 'test.txt'
    if os.access(outfile, 0): os.remove(outfile)

    #labels, distance = get_distance( 'lifts.csv')
    labels, distance = get_from_list('iris_proc')
    #raise()
    #labels, distance = get_from_temp('temp1')
    testtree = optics.setOfObjects(labels, distance)
    testtree.prep_optics(epsilon, minpts)
    testtree.build_optics(epsilon, minpts, outfile)




    f = p.read_csv(outfile, names=['i', 'rd'])
    rd = f['rd']
    pl.plot(rd)
    pl.title('epsilon = ' + str(epsilon) + ', minpts = ' + str(minpts))
    pl.show()
