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
    def __init__(self, px, py, theta, vel):
        self.dt = 1.0 / FPS
        
        self.X = np.array([px, py, theta, vel])
        
        # Measurement uncertainty
        varTheta = 0.05
        varVel = 1.0
        self.R = np.diag([varTheta**2, varVel**2])

        
        # Prediction uncertainty
        acc_ext_max = 10.0  
        sPos = 0.5*acc_ext_max*self.dt**2
        sTheta = 10 * np.pi / 180
        sVel = acc_ext_max * self.dt
        self.Q = np.diag([sPos**2, sPos**2, sTheta**2, sVel**2])
        

        self.H = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])


        self.P = np.diag([1.0, 1.0, 1.0, 1.0]) * 0.1
        
        self.abs_pos_measurements = []
        

    def predict(self, acc, delta_theta):
        print("before : ", self.X)
        print("delta_theta = ", delta_theta)
        
        c = np.cos(self.X[2])
        s = np.sin(self.X[2])
        v = self.X[3]
        
        self.X[0] = self.X[0] + c * v * self.dt + 0.5 * c * acc * self.dt**2
        self.X[1] = self.X[1] + s * v * self.dt + 0.5 * s * acc * self.dt**2
        self.X[2] = self.X[2] + delta_theta
        self.X[3] = self.X[3] + acc * self.dt
        
        
        # Jacobian of g(x, u)
        G = np.array([
            [1.0, 0.0, -s * (v*self.dt + 0.5*acc*self.dt**2), c*self.dt],
            [0.0, 1.0,  c * (v*self.dt + 0.5*acc*self.dt**2), s*self.dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        print("G\n",G)
        
        # print("A: ", self.A.shape)
        print("Q: ", self.Q.shape)
        print("P: ", self.P.shape)
        print("P:\n", self.P)
        
        # self.P = self.A @ self.P @ self.A.T + self.Q
        self.P = G @ self.P @ G.T + self.Q
        
        print("After :", self.X)
        print("\n\n")

        
    def correct(self, vel, theta):
        measurements = [theta, vel]
        for abs_pos in self.abs_pos_measurements:
            measurements.extend(abs_pos)
        measurements = np.array(measurements)
        print("measurements: ", measurements)
        
        # Build H
        H = self.H
        R = self.R
        if len(self.abs_pos_measurements) > 0:
            new_lines = np.vstack([np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0]
            ])] * len(self.abs_pos_measurements))
            H = np.vstack((H, new_lines))
            
            # Adapt R
            varCP = 0.1
            R = np.eye(H.shape[0]) * varCP**2
            n = self.H.shape[0]
            R[:n, :n] = self.R
        
      
        print("R: ", self.R.shape)
        print("H: ", self.H.shape)
        print("P: ", self.P.shape)
        
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        z = measurements - H.dot(self.X)
        
        self.X = self.X + K @ z
        self.P = self.P - K @ H @ self.P
        self.abs_pos_measurements = []
        
        
    def add_abs_pos_measurement(self, pos):
        self.abs_pos_measurements.append(pos)
        
    
        

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
        ## no command (only motion model)
        self.kf.predict(0.0, 0.0)
        # self.kf.predict(msg.x, msg.y) # with command


    def observe_control_points_callback(self, msg):
        for i in range(len(msg.data)//4):
            meas = [msg.data[i*4], msg.data[i*4+1]]
            cp_pos = [msg.data[i*4+2], msg.data[i*4+3]]
            self.kf.add_abs_pos_measurement([cp_pos[0] - meas[0], cp_pos[1] - meas[1]])
            
    def observe_landmarks_callback(self, msg):
        return 
        for i in range(len(msg.data)//3):
            meas = [msg.data[i*3], msg.data[i*3+1]]
            idx = int(msg.data[i*3+2])
            self.kf.add_landmark_measurement(idx, meas)
        
        
        

    def velocity_callback(self, msg: Vector3) -> None:
        print("velo callback")
        self.kf.correct(msg.x, msg.y)
        
        x, y = self.kf.X[:2]
        
        # Example of publishing acceleration command to Flyappy
        self.x += (msg.x * np.cos(msg.y)) / FPS
        self.y += (msg.x * np.sin(msg.y)) / FPS
        self.pub_pos_integration_.publish(Vector3(self.x, self.y, 0))
        
        
        array = Float64MultiArray()
        n = self.kf.X.shape[0]
        # inv_mapping = np.array([[b, a] for a, b in self.kf.landmark_index_to_index.items()])
        array.data = [n] + self.kf.X.flatten().tolist() + self.kf.P.flatten().tolist() #+ inv_mapping.flatten().tolist()
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
