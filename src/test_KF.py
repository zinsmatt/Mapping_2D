import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x = np.array([[0.0, 0.0, 0.0, 0.0]]).T
P = np.diag([1000.0, 1000.0, 1000.0, 1000.0])


def display_cov(P):
    fig = plt.figure(figsize=(6, 6))
    im = plt.imshow(P, interpolation="none", cmap=plt.get_cmap('binary'))
    plt.title('Initial Covariance Matrix $P$')
    ylocs, ylabels = plt.yticks()
    # set the locations of the yticks
    plt.yticks(np.arange(7))
    # set the locations and labels of the yticks
    plt.yticks(np.arange(4),('$x$', '$y$', '$\dot x$', '$\dot y$'), fontsize=22)

    xlocs, xlabels = plt.xticks()
    # set the locations of the yticks
    plt.xticks(np.arange(7))
    # set the locations and labels of the yticks
    plt.xticks(np.arange(4),('$x$', '$y$', '$\dot x$', '$\dot y$'), fontsize=22)

    plt.xlim([-0.5,3.5])
    plt.ylim([3.5, -0.5])

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    plt.colorbar(im, cax=cax);
    plt.show()


# display_cov(P)

dt = 0.1

# Motion model (Constant Velocity)
A = np.array([
    [1.0, 0.0, dt, 0.0],
    [0.0, 1.0, 0.0, dt],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])


# Measurement model
H = np.array([
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])


# Measurement noise covariance
ra = 10.0**4
R = np.diag([ra, ra])

# Process noise covariance
# external element can influence the car and lead to an acceleration (ex: wind)

sv = 8.8

Q = np.array([
    [(dt**4)/4, 0.0, (dt**3)/2, 0.0],
    [0.0, (dt**4)/4, 0.0, (dt**3)/2],
    [(dt**3)/2, 0.0, dt**2, 0.0],
    [0.0, (dt**3)/2, 0.0, dt**2]
]) * sv**2


# display_cov(Q)




# Simulate measurements
m = 200
vx = 20
vy = 10
noise_strengh = 1

mx = np.array(vx+np.random.randn(m) * noise_strengh)
my = np.array(vy+np.random.randn(m) * noise_strengh)

measurements = np.vstack((mx, my))

print("Measurement std = ", np.std(mx))
print("You assumed %.2f" % R[0, 0])

# fig = plt.figure(figsize=(16,5))
# plt.step(range(m),mx, label='$\dot x$')
# plt.step(range(m),my, label='$\dot y$')
# plt.ylabel(r'Velocity $m/s$')
# plt.title('Measurements')
# plt.legend(loc='best',prop={'size':18})
# plt.show()


# Preallocation for Plotting
xt = []
yt = []
dxt= []
dyt= []
Zx = []
Zy = []
Px = []
Py = []
Pdx= []
Pdy= []
Rdx= []
Rdy= []
Kx = []
Ky = []
Kdx= []
Kdy= []

def savestates(x, Z, P, R, K):
    xt.append(float(x[0]))
    yt.append(float(x[1]))
    dxt.append(float(x[2]))
    dyt.append(float(x[3]))
    Zx.append(float(Z[0]))
    Zy.append(float(Z[1]))
    Px.append(float(P[0,0]))
    Py.append(float(P[1,1]))
    Pdx.append(float(P[2,2]))
    Pdy.append(float(P[3,3]))
    Rdx.append(float(R[0,0]))
    Rdy.append(float(R[1,1]))
    Kx.append(float(K[0,0]))
    Ky.append(float(K[1,0]))
    Kdx.append(float(K[2,0]))
    Kdy.append(float(K[3,0]))   
  
  
    
# x[2, 0] = vx
# x[3, 0] = vy



x_gt = x[:2, :].flatten()
gt_pos = [x_gt]

x_noisy = x_gt.copy()
noisy_pos = [x_noisy]

# GT and noisy integration
for i in range(m):
    x_gt[0] += vx * dt
    x_gt[1] += vy * dt
    gt_pos.append(x_gt.tolist()) 
    
    x_noisy[0] += measurements[0, i] * dt
    x_noisy[1] += measurements[1, i] * dt
    noisy_pos.append(x_noisy.tolist())
    
gt_pos = np.vstack(gt_pos)
noisy_pos = np.vstack(noisy_pos)


pred_pos = [x_gt.copy()]
filt_pos = [x_gt.copy()]

    
for i in range(3):

    # Prediction
    x = A @ x
    P = A @ P @ A.T + Q
    
    print("\n\n", i)
    print("Predicted x : ", x.T)
    print("Predicted P \n", P)
    
    pred_pos.append(x[:2, :].flatten().tolist())
    
    # Correction (Measurement Update)
    # Kalman gain
    S = H @ P @ H.T + R
    K = (P @ H.T) @ np.linalg.inv(S)
    
    z = measurements[:, i].reshape((2, 1))
    y = z - H @ x
    
    print("measurements: ", z.T)
    
    print("y = \n", y)
    print("K = \n", K)
    
    x = x + K @ y
    print("Corected x: ", x.T)
    
    P = (np.eye(4) - (K @ H)) @ P
    # P = P - K @ H @ P
    filt_pos.append(x[:2, :].flatten().tolist())
    
    
    savestates(x, z, P, R, K)
    
filt_pos = np.vstack(filt_pos)
pred_pos = np.vstack(pred_pos)

# def plot_K():
#     fig = plt.figure(figsize=(16,9))
#     plt.plot(range(len(measurements[0])),Kx, label='Kalman Gain for $x$')
#     plt.plot(range(len(measurements[0])),Ky, label='Kalman Gain for $y$')
#     plt.plot(range(len(measurements[0])),Kdx, label='Kalman Gain for $\dot x$')
#     plt.plot(range(len(measurements[0])),Kdy, label='Kalman Gain for $\dot y$')

#     plt.xlabel('Filter Step')
#     plt.ylabel('')
#     plt.title('Kalman Gain (the lower, the more the measurement fullfill the prediction)')
#     plt.legend(loc='best',prop={'size':22})
#     plt.show()

# plot_K()


def plot_P():
    fig = plt.figure(figsize=(16,9))
    plt.plot(range(len(measurements[0])),Px, label='$x$')
    plt.plot(range(len(measurements[0])),Py, label='$y$')
    plt.plot(range(len(measurements[0])),Pdx, label='$\dot x$')
    plt.plot(range(len(measurements[0])),Pdy, label='$\dot y$')

    plt.xlabel('Filter Step')
    plt.ylabel('')
    plt.title('Uncertainty (Elements from Matrix $P$)')
    plt.legend(loc='best',prop={'size':22})
    plt.show()
    
# plot_P()



def plot_x():
    fig = plt.figure(figsize=(16,9))
    plt.step(range(len(measurements[0])),dxt, label='$\dot x$')
    plt.step(range(len(measurements[0])),dyt, label='$\dot y$')

    plt.axhline(vx, color='#999999', label='$\dot x_{real}$')
    plt.axhline(vy, color='#999999', label='$\dot y_{real}$')

    plt.xlabel('Filter Step')
    plt.title('Estimate (Elements from State Vector $x$)')
    plt.legend(loc='best',prop={'size':22})
    plt.ylim([0, 30])
    plt.ylabel('Velocity')
    plt.show()

# plot_x()



# plt.figure("Position")
# plt.scatter(gt_pos[:, 0], gt_pos[:, 1], label="gt")
# plt.scatter(noisy_pos[:, 0], noisy_pos[:, 1], label="noisy")
# plt.scatter(pred_pos[:, 0], pred_pos[:, 1], label="pred")
# plt.scatter(filt_pos[:, 0], filt_pos[:, 1], label="filt")
# plt.legend()
# plt.show()