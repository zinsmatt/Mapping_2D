import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Bool
import functools


print = functools.partial(print, flush=True)


FPS = 30

class KF:
    def __init__(self, px, py, vx, vy):
        self.dt = 1.0 / FPS
        self.u = np.array([])
        self.A = np.array([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0, self.dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        self.B = np.array([
            [0.5*self.dt**2, 0.0],
            [0.0, 0.5*self.dt**2],
            [self.dt, 0.0],
            [0.0, self.dt]
        ])

        
        self.X = np.array([px, py, vx, vy])
        self.init_measurements()
        
        # Measurement uncertainty
        ra = 1
        # self.R = np.eye(2) * ra
        self.meas_cov = [ra, ra]

        
        # Prediction uncertainty        
        sv = 0.8
        self.Q = np.array([
            [(self.dt**4)/4, 0.0, (self.dt**3)/2, 0.0],
            [0.0, (self.dt**4)/4, 0.0, (self.dt**3)/2],
            [(self.dt**3)/2, 0.0, self.dt**2, 0.0],
            [0.0, (self.dt**3)/2, 0.0, self.dt**2]
        ]) * sv**2

        self.P = np.diag([1.0, 1.0, 1.0, 1.0])
        
        self.abs_pos_meas = []
        self.landmark_rel_meas = []
        
        self.landmark_index_to_index = {}


    def init_measurements(self):
        self.abs_pos_meas = []
        self.H = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        # Measurement uncertainty
        ra = 1
        self.meas_cov = [ra, ra]
        
        
    
    def predict(self, acc_x, acc_y):
        print("before : ", self.X)
        u = np.array([acc_x, acc_y]) # command
        self.X = self.A.dot(self.X) + self.B.dot(u)
        
        print("A: ", self.A.shape)
        print("Q: ", self.Q.shape)
        print("P: ", self.P.shape)
        self.P = self.A @ self.P @ self.A.T + self.Q
        
        
        print("After :", self.X)
        print("\n\n")

        
    def correct(self, vx, vy):
        measurements = [vx, vy] + np.asarray(self.landmark_rel_meas).flatten().tolist()
        measurements += self.abs_pos_meas
        measurements = np.array(measurements)
        print("measurements: ", measurements)
        
        size_cp = len(self.abs_pos_meas)
        # build R
        rcp = 1.0
        R = np.diag(self.meas_cov + [rcp] * size_cp)
        # build H
        new_lines = np.zeros((size_cp, self.H.shape[1]))
        new_lines[0::2, 0] = 1
        new_lines[1::2, 1] = 1
        H = np.vstack((self.H, new_lines))
        
        print("R: ", R.shape)
        print("H: ", H.shape)
        print("P: ", self.P.shape)
        
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        z = measurements - H.dot(self.X)
        
        self.X = self.X + K @ z
        self.P -= K @ H @ self.P
        self.abs_pos_meas = []
        self.reinit_landmarks_cov()
        
        # print("P:\n", self.P)
        
        
        
    def add_abs_pos_measurement(self, pos):
        self.abs_pos_meas.extend(pos)
        # rcp = 1.0
        # self.meas_cov.extend([rcp, rcp])
        # new_lines = np.zeros((2, self.H.shape[1]))
        # new_lines[0, 0] = 1
        # new_lines[1, 1] = 1
        # self.H = np.vstack((self.H, new_lines))
        
    def reinit_landmarks_cov(self):
        unknown = 100000
        for i in range(2, len(self.meas_cov)):
            self.meas_cov[i] = unknown

        
    
    def add_landmark_measurement(self, index, rel_pos):
        print("################ ADd landmark ", index)
        rl = 1.0

        if index not in self.landmark_index_to_index.keys(): # first obs
            print("new one")
            # insert into state
            x = rel_pos[0] + self.X[0]
            y = rel_pos[1] + self.X[1]
            self.X = np.hstack((self.X, [x, y]))
            
            # update P
            n = self.P.shape[0]
            rp = 1000
            temp = np.eye(n+2) * rp
            temp[:n, :n] = self.P
            self.P = temp
            
            
            # update A
            n = self.A.shape[0]
            temp = np.eye(n+2)
            temp[:n, :n] = self.A
            self.A = temp
            
            # update B
            self.B = np.vstack((self.B, np.zeros((2, 2))))
            
            # update R
            self.meas_cov.extend([rl, rl])
            
            
            #update Q
            n = self.Q.shape[0]
            temp = np.zeros((n+2, n+2))
            temp[:n, :n] = self.Q
            self.Q = temp
            
            #update H
            m, n = self.H.shape
            temp = np.zeros((m+2, n+2))
            temp[:m, :n] = self.H
            self.H = temp
            self.H[m, 0] = -1
            self.H[m, n] = 1
            self.H[m+1, 1] = -1
            self.H[m+1, n+1] = 1
            
            new_idx = len(self.landmark_rel_meas)
            self.landmark_rel_meas.append(rel_pos)
            self.landmark_index_to_index[index] = new_idx
        else:
            print("update old one")
            idx = self.landmark_index_to_index[index]
            self.landmark_rel_meas[idx] = rel_pos
            self.meas_cov[2+idx*2] = rl
            self.meas_cov[2+idx*2+1] = rl
            
        
    
        

class PerceptionRos:
    def __init__(self):
        # Publisher for sending acceleration commands to Flyappy
        self.pub_pos_integration_ = rospy.Publisher(
            "/estimation_pos_integration",
            Vector3,
            queue_size=10
        )
        
        self.pub_pos_kf_ = rospy.Publisher(
            "/estimation_pos_kf",
            Float64MultiArray,
            queue_size=10
        )

        self.sub_vel_ = rospy.Subscriber(
            "/perception_vel",
            Vector3,
            self.velocity_callback
        )
        self.sub_acc_cmd_ = rospy.Subscriber(
            "/acc_cmd",
            Vector3,
            self.acc_cmd_callback
        )
        self.sub_control_points = rospy.Subscriber(
            "/cp_obs",
            Float64MultiArray,
            self.observe_control_points_callback
        )
        self.sub_landmarks_ = rospy.Subscriber(
            "/landmarks_obs",
            Float64MultiArray,
            self.observe_landmarks_callback
        )
        self.sub_quit_ = rospy.Subscriber(
            "/perception_quit",
            Bool,
            self.quit
        )

        self.x = 0.0
        self.y = 0.0
        self.kf = None
        self.kf = KF(0.0, 0.0, 0.0, 0.0)
        self.meas = []
    
    def acc_cmd_callback(self, msg: Vector3) -> None:
        self.kf.predict(msg.x, msg.y)


    def observe_control_points_callback(self, msg):
        for i in range(len(msg.data)//4):
            meas = [msg.data[i*4], msg.data[i*4+1]]
            cp_pos = [msg.data[i*4+2], msg.data[i*4+3]]
            self.kf.add_abs_pos_measurement([cp_pos[0]-meas[0], cp_pos[1]-meas[1]])
            
    def observe_landmarks_callback(self, msg):
        for i in range(len(msg.data)//3):
            meas = [msg.data[i*3], msg.data[i*3+1]]
            idx = int(msg.data[i*3+2])
            self.kf.add_landmark_measurement(idx, meas)
        
        
        

    def velocity_callback(self, msg: Vector3) -> None:
        print("velo callback")
        self.kf.correct(msg.x, msg.y)
        
        x, y = self.kf.X[:2]
        
        # Example of publishing acceleration command to Flyappy
        self.x += msg.x / FPS
        self.y += msg.y / FPS
        
        self.pub_pos_integration_.publish(Vector3(self.x, self.y, 0))
        array = Float64MultiArray()
        n = self.kf.X.shape[0]
        inv_mapping = np.array([[b, a] for a, b in self.kf.landmark_index_to_index.items()])
        array.data = [n] + self.kf.X.flatten().tolist() + self.kf.P.flatten().tolist() + inv_mapping.flatten().tolist()
        self.pub_pos_kf_.publish(array)
        

    def quit(self, msg: Bool) -> None:
        # rospy.loginfo("Quit.")
        pass


def main() -> None:
    rospy.init_node('perception_node', anonymous=True)
    perception_ros = PerceptionRos()
    rospy.spin()


if __name__ == '__main__':
    main()
