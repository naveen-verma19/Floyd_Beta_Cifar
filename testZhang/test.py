import numpy as np
from sklearn.neighbors import NearestNeighbors
a=np.load('pts_in_hull.npy') #313 x 2    the 313 a,b pairs
#prior = 313x1 probability values
print(a)
def na():  # shorthand for new axis
    return np.newaxis

# points=[[0,0],[1,1],[0,1],[1,0]]
# points=np.array(points) #4x2
# neigh = NearestNeighbors(n_neighbors=2).fit(points) #n_neighbors = Number of neighbors to use by default for kneighbors queries.
# dists,inds=neigh.kneighbors([[0,0.7],[0.6,0],[0,0]])
# wts = np.exp(-dists ** 2 / (2 * 5 ** 2))
# s=np.sum(wts, axis=1)
# s=s[:,na()]
# wts = wts / s
#
#
#
#
# p_inds = np.arange(0, 160, dtype='int')
# p_inds=p_inds[:, na()]
# print("")

pts= np.arange(0,16,dtype='int')
pts.reshape((2,2,2,2))
print("")