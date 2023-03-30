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

import matplotlib.pyplot as plt



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

est_landmarks = {}
est_landmarks_cov = {}

pos_est_error_integ = []
pos_est_error_kf = []

est_pos_integ = None
est_pos = None
est_pos_cov = None

def shutdownHook():
    print("Quitting")
    pygame.quit()
    sys.exit()

def print_errors():
    global pos_est_error_integ, pos_est_error_kf
    plt.plot(pos_est_error_integ, label="integration")
    plt.plot(pos_est_error_kf, label="kf")
    plt.legend()
    plt.show()

def position_estimation_integration_callback(msg):
    global est_pos_integ
    est_pos_integ = [msg.x, msg.y]
    pos = create_point(msg.x, msg.y)
    last_pos = est_trajectory_integation[-1]
    v = np.asarray(pos) - np.asarray(last_pos)
    if v.dot(v) > 0.1**2:
        est_trajectory_integation.append(pos)


def position_estimation_kf_callback(msg):
    global est_pos, est_pos_cov
    
    data = msg.data
    n = int(data[0])
    state = data[1:1+n]
    pos = create_point(state[0], state[1])
    state_landmarks = np.asarray(state[4:]).reshape(-1, 2).tolist()
    
    P = np.asarray(data[1+n:1+n+n*n]).reshape(n, n)
    
    landmark_index_mapping = data[1+n+n*n:]
    print("landmark index mappgin", landmark_index_mapping, len(landmark_index_mapping))
    to_landmark_index = {}
    for i in range(0, len(landmark_index_mapping), 2):
        to_landmark_index[landmark_index_mapping[i]] = landmark_index_mapping[i+1]
    
    for i, l in enumerate(state_landmarks):
        idx = to_landmark_index[i]
        est_landmarks[idx] = l
        est_landmarks_cov[idx] = P[4+2*i:4+2*i+2, 4+2*i:4+2*i+2]
    
    
    last_pos = est_trajectory_kf[-1]
    v = np.asarray(pos) - np.asarray(last_pos)
    if v.dot(v) > 0.1**2:
        est_trajectory_kf.append(pos)
    
    est_pos = state[:2]
    est_pos_cov = P[:2, :2]


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
    pub_cp_obs = rospy.Publisher("cp_obs", Float64MultiArray, queue_size=10)
    pub_landmark_obs = rospy.Publisher("landmarks_obs", Float64MultiArray, queue_size=10)
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
    robot.theta = 0#-np.pi / 2
    robot.velocity = 0
    est_trajectory_integation.append([ORIGIN_X, ORIGIN_Y])
    est_trajectory_kf.append([ORIGIN_X, ORIGIN_Y])


    # control_points = [(70, 40), (30, 30), (80, 20), (10, 80)]
    # control_points = [(20, 20), (-30, 30), (-35, 20), (10, -35)]
    # control_points = np.vstack((np.random.uniform(-10, XMAX, size=NUM_CONTROL_POINTS),
    #                      np.random.uniform(-100, 100, size=NUM_CONTROL_POINTS))).T
    # control_points = np.vstack((np.random.uniform(-100, 100, size=NUM_CONTROL_POINTS),
    #                      np.random.uniform(-100, 100, size=NUM_CONTROL_POINTS))).T
    
    control_points = np.vstack((np.random.uniform(-50, 50, size=NUM_CONTROL_POINTS),
                                np.random.uniform(-50, 50, size=NUM_CONTROL_POINTS))).T
    
    landmarks = np.vstack((np.random.uniform(-50, 50, size=NUM_LANDMARKS),
                           np.random.uniform(-50, 50, size=NUM_LANDMARKS))).T
    
    
    

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
        
        acc_x = 0.0
        acc_y = 0.0
        acc = 0.0
        theta_inc = 0
        
        ACC_INCR = 10
        if pressed[pygame.K_UP]:
            acc = ACC_INCR
        if pressed[pygame.K_DOWN]:
            acc = -ACC_INCR
        if pressed[pygame.K_LEFT]:
            theta_inc = THETA_INCR
        if pressed[pygame.K_RIGHT]:
            theta_inc = -THETA_INCR
        
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pub_quit.publish(Bool(True))
                print_errors()
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

            # if event.type == KEYDOWN and (event.key == K_UP):
            #     robot.vy += VELOCITY_INCR
            # if event.type == KEYDOWN and (event.key == K_DOWN):
            #     robot.vy -= VELOCITY_INCR
            # if event.type == KEYDOWN and (event.key == K_LEFT):
            #     robot.vx -= VELOCITY_INCR
            # if event.type == KEYDOWN and (event.key == K_RIGHT):
            #     robot.vx += VELOCITY_INCR
                
            # ACC_INCR = 10
            # if event.type == KEYDOWN and (event.key == K_UP):
            #     acc = ACC_INCR
            # if event.type == KEYDOWN and (event.key == K_DOWN):
            #     acc = -ACC_INCR
            # if event.type == KEYDOWN and (event.key == K_LEFT):
            #     theta_inc = THETA_INCR
            # if event.type == KEYDOWN and (event.key == K_RIGHT):
            #     theta_inc = THETA_INCR
                
                
        

        # tx, ty = targets[target_i]
        # if (robot.x - tx)**2 + (robot.y - ty)**2 < 0.1**2:
        #     target_i = (target_i+1) % len(targets)
        
        
        # control acc
        # dvx = limit_vel(tx-robot.x) - robot.vx
        # dvy = limit_vel(ty-robot.y) - robot.vy
        # acc_x = dvx*FPS
        # acc_y = dvy*FPS      
        
        
         
        
        
        # print("Robot ", robot.x, robot.y, " || ", robot.vx, robot.vy)
        print("Robot ", robot.x, robot.y, " || ", robot.theta, robot.velocity)
        
        c=np.cos(robot.theta)
        s=np.sin(robot.theta)
        robot.x += 0.5 * acc * c / FPS**2 + c * robot.velocity / FPS
        robot.y += 0.5 * acc * s / FPS**2 + s * robot.velocity / FPS
        # robot.x += 0.5 * acc_x / FPS**2 + robot.vx / FPS
        # robot.y += 0.5 * acc_y / FPS**2 + robot.vy / FPS
        
        robot.velocity += acc / FPS
        robot.theta += theta_inc
        
        # robot.vx += acc_x / FPS
        # robot.vy += acc_y / FPS
        # d = robot.velocity / FPS
        # robot.x += np.cos(robot.theta) * d
        # robot.y += np.sin(robot.theta) * d
        
        
        
        # Measure control points
        control_points_meas = []
        for a in control_points:
            if robot.is_visible(a):
                # simulate measurement 
                # we measure landmark relative to robot + we know landmarks position => we can get the robot position
                control_points_meas.extend([a[0]-robot.x, a[1]-robot.y, a[0], a[1]])
            
        cp_meas_pub = Float64MultiArray()
        cp_meas_pub.data = control_points_meas
        pub_cp_obs.publish(cp_meas_pub)        
        
        
        # Measure landmarks        
        landmarks_meas = []
        for i, a in enumerate(landmarks):
            if robot.is_visible(a):
                landmarks_meas.extend([a[0]-robot.x, a[1]-robot.y, i]) # dx, dy, idx
        landmarks_meas_pub = Float64MultiArray()
        landmarks_meas_pub.data = landmarks_meas
        pub_landmark_obs.publish(landmarks_meas_pub)
            
        
        
        
        
        
        
        
        
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
        
        
        if est_pos_cov is not None:
            axes, angle = ellipes_from_cov(est_pos_cov, 0.99)
            ell_pts = create_ellipse(est_pos[0], est_pos[1], axes[0], axes[1], angle)
            pygame.draw.polygon(SCREEN, (255, 255, 255), shift(ell_pts, -dx, -dy), width=1)
            
            
        
        
        for a in control_points:
            square_pts = create_square_polygon(*a, ANCHORS_SIZE)
            col = ANCHORS_COL
            if robot.is_visible(a):
                col = ANCHORS_VISIBLE_COL
            pygame.draw.polygon(SCREEN, col, shift(square_pts, -dx, -dy))

        for a in landmarks:
            circle = create_circle(*a, ANCHORS_SIZE)
            col = ANCHORS_COL
            if robot.is_visible(a):
                col = ANCHORS_VISIBLE_COL
            pygame.draw.circle(SCREEN, col, shift(circle["center"], -dx, -dy), circle["radius"])
            
            
        for idx, a in est_landmarks.items():
            circle = create_circle(*a, ANCHORS_SIZE)
            col = (0, 0, 255)
            pygame.draw.circle(SCREEN, col, shift(circle["center"], -dx, -dy), circle["radius"])
            
            axes, angle = ellipes_from_cov(est_landmarks_cov[idx], 0.99)
            ell_pts = create_ellipse(a[0], a[1], axes[0], axes[1], angle)
            pygame.draw.polygon(SCREEN, (255, 255, 255), shift(ell_pts, -dx, -dy), width=1)
            

        # Send command
        # noisy_acc_cmd_x, noisy_acc_cmd_y = add_gaussian_noise([acc_x, acc_y], ACC_NOISE_SCALE)
        # noisy_acc_cmd, _ = add_gaussian_noise([acc, 0], ACC_NOISE_SCALE)
        # noisy_delta_theta_cmd = add_gaussian_noise([theta_inc], 2*np.pi/180)[0]
        # pub_acc_cmd.publish(Vector3(noisy_acc_cmd, noisy_delta_theta_cmd, 0))
        pub_acc_cmd.publish(Vector3(0, 0, 0)) # do not send command but just publish to do predictios
        
        
        # noisy_vx, noisy_vy = add_gaussian_noise([robot.vx, robot.vy], VELOCITY_NOISE_SCALE)
        noisy_vel, _ = add_gaussian_noise([robot.velocity, 0], 2*VELOCITY_NOISE_SCALE)
        noisy_theta = add_gaussian_noise([robot.theta], (5*np.pi)/180)[0]
        pub_velocity.publish(Vector3(noisy_vel, noisy_theta, 0))
        
        if est_pos is not None and est_pos_integ is not None:
            pos_est_error_kf.append(distance(est_pos, [robot.x, robot.y]))
            pos_est_error_integ.append(distance(est_pos_integ, [robot.x, robot.y]))
        
        
        SCREEN.blit(ALPHA_SCREEN, (0, 0))
        pygame.display.update()
        FPSCLOCK.tick(FPS)
    
    
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    