import numpy as np
import pygame.surfarray as surfarray
import pygame
from pygame.locals import *
import sys
import math
import rospkg

import rospy
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool
from std_msgs.msg import Float64MultiArray

from draw import *
from constants import *
from utils import *


def ellipes_from_cov(P, confidence):
    Pxy = P[:2, :2]
    eigvals, R = np.linalg.eig(Pxy)
    print(eigvals)
    
    d = np.sqrt(-2 * np.log(1-confidence))
    
    return d * np.sqrt(eigvals), np.arctan2(R[1, 0], R[0, 0])

XMAX = 5000

# targets = [(XMAX, 0), (-30, -30), (30, -30), (30, 30), (-30, 30)]
targets = [(-30, -30), (30, -30), (30, 30), (-30, 30)]
target_i = 0

est_trajectory_integation = []
est_trajectory_kf = []
gt_trajectory = []

ellipse = None

def shutdownHook():
    print("Quitting")
    pygame.quit()
    sys.exit()


def position_estimation_integration_callback(msg):
    global SCREEN
    pos = create_point(msg.x, msg.y)
    last_pos = est_trajectory_integation[-1]
    v = np.asarray(pos) - np.asarray(last_pos)
    if v.dot(v) > 0.1**2:
        est_trajectory_integation.append(pos)


def position_estimation_kf_callback(msg):
    global SCREEN
    data = msg.data
    pos = create_point(data[0], data[1])
    P = np.asarray(msg.data[2:]).reshape(4, 4)
    
    last_pos = est_trajectory_kf[-1]
    v = np.asarray(pos) - np.asarray(last_pos)
    if v.dot(v) > 0.1**2:
        est_trajectory_kf.append(pos)
    
    global ellipse
    ellipse = ellipes_from_cov(P, 0.95)
    print(ellipse)



def add_gaussian_noise(arr, scale):
    noise = np.random.normal(0, scale, len(arr))
    return [x + eps for x, eps in zip(arr, noise)]



def limit_vel(v):
    if v < -VMAX:
        v = -VMAX
    if v > VMAX:
        v = VMAX
    return v




class Robot:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.theta = 0
        self.velocity = 0
        self.vx = 0
        self.vy = 0
        self.fov_range = FOV_RANGE
        self.fov_half_angle = FOV_HALF_ANGLE
    

    def is_visible(self, point):
        pt = np.array([point[0] - self.x, point[1] - self.y])
        
        if pt.dot(pt) > self.fov_range**2:
            return False

        R = get_R(-self.theta)
        pt = R.dot(pt)
        
        a = np.arctan2(pt[1], pt[0])
        if a < -self.fov_half_angle or a > self.fov_half_angle:
            return False
        return True


def shift(arr, dx, dy):
    try:
        res = [(x+dx, y+dy) for x, y in arr]
    except:
        res = [arr[0]+dx, arr[1]+dy]
    return res



def main():
    # Initialize ROS
    rospy.init_node("main", anonymous=True)
    rospy.on_shutdown(shutdownHook)


    pub_velocity = rospy.Publisher("perception_vel", Vector3, queue_size=10)
    pub_acc_cmd = rospy.Publisher("acc_cmd", Vector3, queue_size=10)
    pub_quit = rospy.Publisher("perception_quit", Bool, queue_size=2)
    sub_est_pos_integ = rospy.Subscriber("estimation_pos_integration", Vector3, position_estimation_integration_callback)
    sub_est_pos_kf = rospy.Subscriber("estimation_pos_kf", Float64MultiArray, position_estimation_kf_callback)


    global SCREEN, FPSCLOCK
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    ALPHA_SCREEN = SCREEN.convert_alpha()

    pygame.display.set_caption('Mapping 2D')

        
    robot = Robot()
    robot.x = 0
    robot.y = 0
    robot.theta = -np.pi / 2
    est_trajectory_integation.append([ORIGIN_X, ORIGIN_Y])
    est_trajectory_kf.append([ORIGIN_X, ORIGIN_Y])


    # anchors = [(70, 40), (30, 30), (80, 20), (10, 80)]
    # anchors = [(20, 20), (-30, 30), (-35, 20), (10, -35)]
    # anchors = np.vstack((np.random.uniform(-10, XMAX, size=NUM_ANCHORS),
    #                      np.random.uniform(-100, 100, size=NUM_ANCHORS))).T
    anchors = np.vstack((np.random.uniform(-100, 100, size=NUM_ANCHORS),
                         np.random.uniform(-100, 100, size=NUM_ANCHORS))).T

    pygame.key.set_repeat(int(1000/FPS))

    global targets, target_i
    while True:
        SCREEN.fill(BACKGROUND_COLOR)
        ALPHA_SCREEN.fill([0,0,0,0])
        
        pressed = pygame.key.get_pressed()

        # if pressed[pygame.K_UP]:
        #     robot.y += POS_INCR
        # if pressed[pygame.K_DOWN]:
        #     robot.y -= POS_INCR
        # if pressed[pygame.K_LEFT]:
        #     robot.x -= POS_INCR
        # if pressed[pygame.K_RIGHT]:
        #     robot.x += POS_INCR
            
        # if pressed[pygame.K_UP]:
        #     robot.vy += VELOCITY_INCR
        # if pressed[pygame.K_DOWN]:
        #     robot.vy -= VELOCITY_INCR
        # if pressed[pygame.K_LEFT]:
        #     robot.vx -= VELOCITY_INCR
        # if pressed[pygame.K_RIGHT]:
        #     robot.vx += VELOCITY_INCR
        
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pub_quit.publish(Bool(True))
                pygame.quit()
                sys.exit()
            # # if event.type == KEYDOWN and (event.key == K_UP):
            # #     robot.velocity += VELOCITY_INCR
            # # if event.type == KEYDOWN and (event.key == K_DOWN):
            # #     robot.velocity -= VELOCITY_INCR
            # # if event.type == KEYDOWN and (event.key == K_LEFT):
            # #     robot.theta += THETA_INCR
            # # if event.type == KEYDOWN and (event.key == K_RIGHT):
            # #     robot.theta -= THETA_INCR

            if event.type == KEYDOWN and (event.key == K_UP):
                robot.vy += VELOCITY_INCR
            if event.type == KEYDOWN and (event.key == K_DOWN):
                robot.vy -= VELOCITY_INCR
            if event.type == KEYDOWN and (event.key == K_LEFT):
                robot.vx -= VELOCITY_INCR
            if event.type == KEYDOWN and (event.key == K_RIGHT):
                robot.vx += VELOCITY_INCR
        

        tx, ty = targets[target_i]
        if (robot.x - tx)**2 + (robot.y - ty)**2 < 0.1**2:
            target_i = (target_i+1) % len(targets)
        
        
        # control acc
        dvx = limit_vel(tx-robot.x) - robot.vx
        dvy = limit_vel(ty-robot.y) - robot.vy
        acc_x = dvx*FPS
        acc_y = dvy*FPS       
        
        
        print("Robot ", robot.x, robot.y, " || ", robot.vx, robot.vy)
        
        
        
        robot.x += 0.5 * acc_x / FPS**2 + robot.vx / FPS
        robot.y += 0.5 * acc_y / FPS**2 + robot.vy / FPS
        
        robot.vx += acc_x / FPS
        robot.vy += acc_y / FPS
        # d = robot.velocity / FPS
        # robot.x += np.cos(robot.theta) * d
        # robot.y += np.sin(robot.theta) * d

        
        
        noisy_acc_cmd_x, noisy_acc_cmd_y = add_gaussian_noise([acc_x, acc_y], ACC_NOISE_SCALE)
        pub_acc_cmd.publish(Vector3(noisy_acc_cmd_x, noisy_acc_cmd_y, 0))
        
        
        noisy_vx, noisy_vy = add_gaussian_noise([robot.vx, robot.vy], VELOCITY_NOISE_SCALE)
        pub_velocity.publish(Vector3(noisy_vx, noisy_vy, 0))
        
        
        
        
        
        
        triangle = create_triangle_polygon(robot.x, robot.y, robot.theta, ROBOT_SIZE)
        
        center = np.asarray(triangle).mean(axis=0)
        dx = center[0] - SCREENWIDTH/2
        dy = center[1] - SCREENHEIGHT/2
        
        gt_trajectory.append(center)
        
        fov = create_arc_polygon(robot.x, robot.y, robot.theta, FOV_RANGE, FOV_HALF_ANGLE)
        
        if len(est_trajectory_integation) > 1:
            pygame.draw.lines(SCREEN, EST_TRAJ_INTEG_COLOR, False, shift(est_trajectory_integation, -dx, -dy))

        if len(est_trajectory_kf) > 1:
            pygame.draw.lines(SCREEN, EST_TRAJ_KF_COLOR, False, shift(est_trajectory_kf, -dx, -dy))

        if len(gt_trajectory) > 1:
            pygame.draw.lines(SCREEN, GT_TRAJ_COLOR, False, shift(gt_trajectory, -dx, -dy))

        pygame.draw.polygon(ALPHA_SCREEN, FOV_COLOR, shift(fov, -dx, -dy))
        pygame.draw.polygon(SCREEN, ROBOT_COLOR, shift(triangle, -dx, -dy))
        
        if ellipse is not None:
            axes, angle = ellipse
            ell_pts = create_ellipse(robot.x, robot.y, axes[0], axes[1], angle)
            pygame.draw.polygon(SCREEN, (255, 255, 255), shift(ell_pts, -dx, -dy), width=2)
            
            
        
        
        for a in anchors:
            circle = create_circle(*a, ANCHORS_SIZE)
            col = ANCHORS_COL
            if robot.is_visible(a):
                col = ANCHORS_VISIBLE_COL
            pygame.draw.circle(SCREEN, col, shift(circle["center"], -dx, -dy), circle["radius"])


        SCREEN.blit(ALPHA_SCREEN, (0, 0))
        pygame.display.update()
        FPSCLOCK.tick(FPS)
    
    
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
