import numpy as np

FPS = 30
SCREENWIDTH  = 1000
SCREENHEIGHT = 800
SCALING = 0.1 # m/pixel

ANCHORS_SIZE = 0.5

FOV_RANGE = 20.0
FOV_HALF_ANGLE = np.pi / 4

BACKGROUND_COLOR = (55, 55, 55)

VELOCITY_INCR = 1.0
THETA_INCR = np.pi / 20
POS_INCR = 0.5

ANCHORS_COL = (255, 0, 0)
ANCHORS_VISIBLE_COL = (0, 255, 0)

ROBOT_SIZE = 1

EST_TRAJ_INTEG_COLOR = (255, 0, 255)
EST_TRAJ_KF_COLOR = (0, 255, 0)
GT_TRAJ_COLOR = (255, 255, 255)

ORIGIN_X = SCREENWIDTH / 2
ORIGIN_Y = SCREENHEIGHT / 2

FOV_ALPHA = 55
FOV_COLOR = (3, 141, 255, FOV_ALPHA)

ROBOT_COLOR = (255, 255, 0)

VELOCITY_NOISE_SCALE = 5.0
ACC_NOISE_SCALE = 0.5

NUM_ANCHORS = 3

VMAX = 20
