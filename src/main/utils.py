import numpy as np

def get_R(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]])


def distance(p0, p1):
    return np.sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2)