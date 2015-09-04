#!/usr/bin/python

def get_distance(in_file):
    fh = open(in_file)
    labels = map(lambda x: x.strip('"'), fh.readline().strip().split(','))
    z = len(labels)
    out = {}
    for nr, l in enumerate(fh):
        l = l.strip().split(',')
        for x in range(nr+1, z):
            out[(x,nr)] = l[x]
    return out



if __name__=="__main__":
    distance = get_distance( 'lifts.csv')
