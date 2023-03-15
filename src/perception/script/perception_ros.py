import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Bool

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
        # self.B = np.array([
        #     [0.5*self.dt**2, 0.0],
        #     [0.0, 0.5*self.dt**2],
        #     [self.dt, 0.0],
        #     [0.0, self.dt]
        # ])
        self.B = np.array([
            [0.5*self.dt**2, 0.0],
            [0.0, 0.5*self.dt**2],
            [self.dt, 0.0],
            [0.0, self.dt]
        ])
        self.H = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        self.X = np.array([px, py, vx, vy])
        
        
        # Measurement uncertainty
        ra = 10000
        self.R = np.eye(2) * ra

        
        # Prediction uncertainty        
        sv = 8.8
        self.Q = np.array([
            [(self.dt**4)/4, 0.0, (self.dt**3)/2, 0.0],
            [0.0, (self.dt**4)/4, 0.0, (self.dt**3)/2],
            [(self.dt**3)/2, 0.0, self.dt**2, 0.0],
            [0.0, (self.dt**3)/2, 0.0, self.dt**2]
        ]) * sv**2

        self.P = np.diag([1.0, 1.0, 1.0, 1.0])

    
    def predict(self, acc_x, acc_y):
        
        print("before : ", self.X)
        u = np.array([acc_x, acc_y]) # command
        self.X = self.A.dot(self.X) + self.B.dot(u)
        self.P = self.A @ self.P @ self.A.T + self.Q
        
        
        print("After :", self.X)
        print("\n\n")

        
    def correct(self, vx, vy):
        measurements = np.array([vx, vy])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        z = measurements - self.H.dot(self.X)
        
        self.X = self.X + K @ z
        self.P -= K @ self.H @ self.P
        
        
        

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
        self.sub_quit_ = rospy.Subscriber(
            "/perception_quit",
            Bool,
            self.quit
        )

        self.x = 0.0
        self.y = 0.0
        self.kf = None
        self.kf = KF(0.0, 0.0, 0.0, 0.0)
    
    def acc_cmd_callback(self, msg: Vector3) -> None:
        self.kf.predict(msg.x, msg.y)


    def velocity_callback(self, msg: Vector3) -> None:
        self.kf.correct(msg.x, msg.y)
        
        x, y = self.kf.X[:2]
        
        # Example of publishing acceleration command to Flyappy
        self.x += msg.x / FPS
        self.y += msg.y / FPS
        
        self.pub_pos_integration_.publish(Vector3(self.x, self.y, 0))
        array = Float64MultiArray()
        array.data = [x, y] + self.kf.P.flatten().tolist()
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