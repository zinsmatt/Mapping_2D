import numpy as np
import math
from constants import *
from utils import *

def create_triangle_polygon(x, y, theta, size):
    A = [size, 0.0]
    d = size * 0.7
    l = math.sqrt(size**2 - d**2)
    B = [-d, l]
    C = [-d, -l]
    corners = np.vstack((A, B, C))
    R = get_R(theta)
    corners = (R @ corners.T).T
    corners[:, 0] += x
    corners[:, 1] += y
    
    corners /= SCALING
    corners[:, 1] = -corners[:, 1]
    
    corners[:, 0] += ORIGIN_X
    corners[:, 1] += ORIGIN_Y
    
    return corners.tolist()

def create_arc_polygon(x, y, theta, distance, half_view_angle):
    points = np.array([[0.0, 0.0]])
    angles = np.linspace(-half_view_angle, half_view_angle, int(round(np.rad2deg(half_view_angle*2))))
    new_points = np.vstack((np.cos(angles), np.sin(angles))).T * distance
    points = np.vstack((points, new_points))
    R = get_R(theta)
    points = (R @ points.T).T
    points[:, 0] += x
    points[:, 1] += y
    points /= SCALING
    points[:, 1] = -points[:, 1]
    
    points[:, 0] += ORIGIN_X
    points[:, 1] += ORIGIN_Y
    return points.tolist()


def create_circle(x, y, size):
    return {
        "center": (x / SCALING + ORIGIN_X,  -y / SCALING + ORIGIN_Y),
        "radius": size / SCALING
    }

def create_square_polygon(x, y, size):
    s = size / 2
    corners = np.array([
        [x-s, y-s],
        [x+s, y-s],
        [x+s, y+s],
        [x-s, y+s],
    ])
    corners /= SCALING
    corners[:, 1] = -corners[:, 1]
    
    corners[:, 0] += ORIGIN_X
    corners[:, 1] += ORIGIN_Y
    
    return corners.tolist()


def create_point(x, y):
    return [ORIGIN_X + x / SCALING, ORIGIN_Y - y / SCALING]



def create_ellipse(x, y, ax, ay, angle):
    angles = np.linspace(0, np.pi*2, 720)
    points = np.vstack((np.cos(angles) * ax, np.sin(angles) * ay)).T
    R = get_R(angle)
    points = (R @ points.T).T
    points[:, 0] += x
    points[:, 1] += y
    points /= SCALING
    points[:, 1] = -points[:, 1]
    
    points[:, 0] += ORIGIN_X
    points[:, 1] += ORIGIN_Y
    return points.tolist()