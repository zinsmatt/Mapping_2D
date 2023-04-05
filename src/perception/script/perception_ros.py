import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Bool
import functools
import g2o

print = functools.partial(print, flush=True)


FPS = 30

class KF:
    def __init__(self, px, py, theta, vel):
        self.dt = 1.0 / FPS
        
        self.X = np.array([px, py, theta, vel])
        
        # Measurement uncertainty
        varTheta = np.deg2rad(1.0)
        varVel = 1.0
        # self.R = np.diag([varTheta**2, varVel**2])
        self.meas_cov = [varTheta**2, varVel**2]
        
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
        self.landmark_rel_meas = []
        self.landmark_index_to_index = {}


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
        # self.X[>=4] = constant (landmarks are static)
        

        # Build G        
        # Jacobian of g(x, u)
        G_base = np.array([
            [1.0, 0.0, -s * (v*self.dt + 0.5*acc*self.dt**2), c*self.dt],
            [0.0, 1.0,  c * (v*self.dt + 0.5*acc*self.dt**2), s*self.dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        print("G\n",G_base)
        G = np.eye(self.X.shape[0])
        G[:4, :4] = G_base
        
        
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
        for rel_pos in self.landmark_rel_meas:
            measurements.extend(rel_pos)
        for abs_pos in self.abs_pos_measurements:
            measurements.extend(abs_pos)
        measurements = np.array(measurements)
        print("measurements: ", measurements)
        
        # Build H
        H = self.H
        new_lines = np.zeros((len(self.abs_pos_measurements)*2, self.H.shape[1]))
        new_lines[0::2, 0] = 1
        new_lines[1::2, 1] = 1
        H = np.vstack((self.H, new_lines))
            
        # Adapt R
        varCP = 0.1
        R = np.diag(self.meas_cov + [varCP**2]*len(self.abs_pos_measurements)*2)
        
        
      
        print("R: ", R.shape)
        print("H: ", H.shape)
        print("P: ", self.P.shape)
        
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        z = measurements - H.dot(self.X)
        
        self.X = self.X + K @ z
        self.P = self.P - K @ H @ self.P
        
        
        self.abs_pos_measurements = []
        self.reinit_landmarks_cov()

        
    def reinit_landmarks_cov(self):
        unknown = 100000
        for i in range(2, len(self.meas_cov)):
            self.meas_cov[i] = unknown
        
    def add_abs_pos_measurement(self, pos):
        self.abs_pos_measurements.append(pos)
        
    def add_landmark_measurement(self, index, rel_pos):
        print("################ ADd landmark ", index)
        varLandmark = 0.1

        if index not in self.landmark_index_to_index.keys(): # first obs
            print("new one")
            # insert into state
            x = rel_pos[0] + self.X[0]
            y = rel_pos[1] + self.X[1]
            self.X = np.hstack((self.X, [x, y]))
            
            # update P
            n = self.P.shape[0]
            varLandmark_init = 100
            temp = np.eye(n+2) * varLandmark_init**2
            temp[:n, :n] = self.P
            self.P = temp
            
            # update R
            self.meas_cov.extend([varLandmark**2, varLandmark**2])
            
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
            self.meas_cov[2+idx*2] = varLandmark**2
            self.meas_cov[2+idx*2+1] = varLandmark**2
            


class PoseGraphOptim:
    def __init__(self):
        self.keyframes = []

    def add_keyframe(self, pose):
        self.keyframes.append(pose)
        
    def serialize(self):
        res = [len(self.keyframes)]
        for p in self.keyframes:
            res.extend(p)
        return res       
    
    def optimize(self, final_pos):
        print(self.keyframes)
        print(final_pos)
        optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE2(g2o.LinearSolverCholmodSE2())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer.set_algorithm(solver)
        
        vertices = []
        for i, pose in enumerate(self.keyframes):
            if i  == len(self.keyframes)-1:
                pe2 = g2o.SE2(final_pos[0], final_pos[1], pose[2])
            else:
                pe2 = g2o.SE2(pose)
            v = g2o.VertexSE2()
            v.set_id(i)
            v.set_estimate(pe2)
            if i  == len(self.keyframes)-1 or i == 0:
                v.set_fixed(True)    
            vertices.append(v)
            optimizer.add_vertex(v)

        for i in range(len(self.keyframes)-1):
            p0 = np.asarray(self.keyframes[i])
            p1 = np.asarray(self.keyframes[i+1])
            s0 = g2o.SE2(p0)
            s1 = g2o.SE2(p1)
            T_rel_0_to_1 = s0.inverse() * s1
            
            edge = g2o.EdgeSE2()
            edge.set_vertex(0, vertices[i])
            edge.set_vertex(1, vertices[i+1])
            edge.set_measurement(T_rel_0_to_1)
            edge.set_information(np.identity(3))
            edge.compute_error()
            optimizer.add_edge(edge)

        optimizer.initialize_optimization()
        optimizer.set_verbose(True)
        optimizer.optimize(500)

        for i, v in enumerate(vertices):
            p = v.estimate()
            self.keyframes[i][:2] = p.translation()
            self.keyframes[i][2] = p.rotation().angle()
        
        

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
        
        self.pub_keyframes_ = rospy.Publisher(
            "/keyframes",
            Float64MultiArray,
            queue_size=10
        )
        self.pub_mode_ = rospy.Publisher(
            "/mode",
            Bool,
            queue_size=0
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
        self.it = 0
        self.posegraph = PoseGraphOptim()
        self.previous_cp_obs = 0
        self.loop_closure_detected = False
    
    def acc_cmd_callback(self, msg: Vector3) -> None:
        ## no command (only motion model)
        self.kf.predict(0.0, 0.0)
        # self.kf.predict(msg.x, msg.y) # with command


    def observe_control_points_callback(self, msg):
        if self.previous_cp_obs == 1 and len(msg.data) == 0:
            self.previous_cp_obs = 2
        for i in range(len(msg.data)//4):
            meas = [msg.data[i*4], msg.data[i*4+1]]
            cp_pos = [msg.data[i*4+2], msg.data[i*4+3]]
            self.kf.add_abs_pos_measurement([cp_pos[0] - meas[0], cp_pos[1] - meas[1]])
            if self.previous_cp_obs == 0:
                self.previous_cp_obs = 1
            elif self.previous_cp_obs == 2:
                self.previous_cp_obs = 3

        if self.previous_cp_obs == 3:
            self.pub_mode_.publish(Bool(True))
            # self.send_keyframes()
            self.loop_closure_detected = True
            
    def observe_landmarks_callback(self, msg):
        for i in range(len(msg.data)//3):
            meas = [msg.data[i*3], msg.data[i*3+1]]
            idx = int(msg.data[i*3+2])
            self.kf.add_landmark_measurement(idx, meas)
        
        
        

    def velocity_callback(self, msg: Vector3) -> None:
        print("velo callback")
        
        if self.loop_closure_detected:
            print("TREAT  LOOP CLOSUURE")
            print("asbosulte pos : ", self.kf.abs_pos_measurements)
            measured_pos = self.kf.abs_pos_measurements[-1]
            self.kf.abs_pos_measurements = []
            self.kf.correct(msg.x, msg.y)
            self.posegraph.add_keyframe(self.kf.X[:3].tolist())
            self.posegraph.optimize(measured_pos)
            
            self.send_keyframes()
            
        else:
            self.kf.correct(msg.x, msg.y)
            
            x, y = self.kf.X[:2]
            
            # Example of publishing acceleration command to Flyappy
            self.x += (msg.x * np.cos(msg.y)) / FPS
            self.y += (msg.x * np.sin(msg.y)) / FPS
            self.pub_pos_integration_.publish(Vector3(self.x, self.y, 0))
            
            
            array = Float64MultiArray()
            n = self.kf.X.shape[0]
            inv_mapping = np.array([[b, a] for a, b in self.kf.landmark_index_to_index.items()])
            array.data = [n] + self.kf.X.flatten().tolist() + self.kf.P.flatten().tolist() + inv_mapping.flatten().tolist()
            self.pub_pos_kf_.publish(array)
            
            if self.it % 1 == 0:
                self.posegraph.add_keyframe(self.kf.X[:3].tolist())
        self.it += 1
        
    def send_keyframes(self):
        array = Float64MultiArray()
        array.data = self.posegraph.serialize()
        self.pub_keyframes_.publish(array)        
        

    def quit(self, msg: Bool) -> None:
        # rospy.loginfo("Quit.")
        self.send_keyframes()

def main() -> None:
    rospy.init_node('perception_node', anonymous=True)
    perception_ros = PerceptionRos()
    rospy.spin()


if __name__ == '__main__':
    main()
