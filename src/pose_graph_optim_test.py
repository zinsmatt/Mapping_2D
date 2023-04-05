import numpy as np
import g2o
import matplotlib.pyplot as plt



keyframes = [[-0.04722334345234501, 0.0, -0.010985252152303298], [2.385119465795997, 0.05571090566977821, 0.03274321543753004], [12.817244562365742, 0.1838493449982668, 0.15206281215182813], [22.329731329741012, -5.6874501810430385, -1.4447085894051888], [13.03998951146221, -12.603711798238574, -3.0460678051134615], [-1.1377598897570331, -20.946735367272243, -3.233167228297691], [-8.894487990170015, -6.37016695588847, -4.261394520225661], [-19.1655838784531, 4.022919971220094, -4.668669067849116], [-9.007898675009088, 8.512908196817166, -6.836253099258125], [-13.558524944821228, -4.015315069519519, -9.453923387174823], [-21.79458883974273, 8.820438860715143, -11.704069931915486], [-8.562515630699062, 7.179345381311133, -12.994842588221331], [2.046959027235213, 5.146338245828082, -12.594271512704449]]
final_pos = [10.375885728933964, 4.164961086626866]


optimizer = g2o.SparseOptimizer()
solver = g2o.BlockSolverSE2(g2o.LinearSolverCholmodSE2())
solver = g2o.OptimizationAlgorithmLevenberg(solver)
optimizer.set_algorithm(solver)

vertices = []
for i, pose in enumerate(keyframes):
    if i  == len(keyframes)-1:
        pe2 = g2o.SE2(final_pos[0], final_pos[1], pose[2])
    else:
        pe2 = g2o.SE2(pose)
    v = g2o.VertexSE2()
    v.set_id(i)
    v.set_estimate(pe2)
    if i  == len(keyframes)-1 or i == 0:
        v.set_fixed(True)    
    vertices.append(v)
    optimizer.add_vertex(v)
    
for i in range(len(keyframes)-1):
    p0 = np.asarray(keyframes[i])
    p1 = np.asarray(keyframes[i+1])
    meas = g2o.SE2(p1-p0)
    
    # r = p1[2]-p0[2]
    # R0 = np.array([
    #     [np.cos(p0[2]), -np.sin(p0[2])],
    #     [np.sin(p0[2]), np.cos(p0[2])]
    # ])
    # dt = R0.T.dot(p1[:2]-p0[:2])
    
    

    s0 = g2o.SE2(p0)
    s1 = g2o.SE2(p1)
    T_rel_0_to_1 = s0.inverse()*s1
    
    edge = g2o.EdgeSE2()
    edge.set_vertex(0, vertices[i])
    edge.set_vertex(1, vertices[i+1])
    print("edge between", i, i+1)
    edge.set_measurement(T_rel_0_to_1)
    edge.set_information(np.identity(3))
    edge.compute_error()
    print("=>", edge.error())
    optimizer.add_edge(edge)

optimizer.initialize_optimization()
optimizer.set_verbose(True)
optimizer.optimize(10)





kf_positions_optim = []
for v in vertices:
    p = v.estimate().translation()
    kf_positions_optim.append(p)
kf_positions_optim = np.asarray(kf_positions_optim)


kf_positions = []
for kf in keyframes:
    kf_positions.append(kf[:2])
kf_positions = np.asarray(kf_positions)
D = 50
plt.xlim(-D, D)
plt.ylim(-D, D)
plt.plot(kf_positions[:, 0], kf_positions[:, 1], label="init")
plt.scatter([final_pos[0]], [final_pos[1]], label="final")
plt.plot(kf_positions_optim[:, 0], kf_positions_optim[:, 1], label="optim")
plt.legend()
plt.show()


# g = g2o.SE2([1, 2, 3])
# g = g2o.SE2(1, 2, 3)
# print(g.rotation().angle())
# print(g.translation())


# v1 = g2o.VertexSE2()
# v1.set_estimate(g2o.SE2([0.0, 0.0, 0.0]))
# v2 = g2o.VertexSE2()
# v2.set_estimate(g2o.SE2([10.0, 0.0, 0.0]))
# print(v1)
# print(v2)
# e = g2o.EdgeSE2()
# e.set_vertex(0, v1)
# e.set_vertex(1, v2)
# T = g2o.SE2([9.0, 0.0, 0.0])
# e.set_measurement(T)
# e.compute_error()
# print(e.error())

