
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import math

import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial.transform import Rotation as Rot

# 로봇이 주어진 입력 u를 받으며 움직이고, 고정된 랜드마크(rf_id)로부터 받은 거리 측정값(일부러 노이즈를 끼움)을 이용해 자신의 위치를 추정하는 코드

# Estimation parameter of PF
#Q = np.diag([0.2]) ** 2  # range 측정 error  대각행렬 ???? 오차에 대한 공분산 행렬
Q = np.diag([0.2,0.2]) ** 2
R = np.diag([2.0, np.deg2rad(40.0)]) ** 2  # input error

#  Simulation parameter
#Q_sim = np.diag([0.2]) ** 2 # 사용되는 노이즈 측정 노이즈
Q_sim = np.diag([0.2, 0.2]) ** 2  # shape: (2,2)

R_sim = np.diag([1.0, np.deg2rad(30.0)]) ** 2 # 입력 노이즈

DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]
MAX_RANGE = 20.0  # maximum observation range 이 거리 내에 있는 랜드마크들만 측정이 되는 것임

# Particle filter parameter
NP = 100  # Number of Particle
NTh = NP / 2.0  # Number of particle for re-sampling

show_animation = True

def rot_mat_2d(angle): # 회전행렬
    """
    Create 2D rotation matrix from an angle

    Parameters
    ----------
    angle :

    Returns
    -------
    A 2D rotation matrix

    Examples
    --------
    >>> angle_mod(-4.0)


    """
    return Rot.from_euler('z', angle).as_matrix()[0:2, 0:2] # 회전축: Z축 회전행렬의 0:2,0:2 추출 -> ????

def calc_input():  # 입력 U 정의
    v = 1.0  # [m/s]
    yaw_rate = 0.1  # [rad/s]
    u = np.array([[v, yaw_rate]]).T
    return u

 
# def observation(x_true, xd, u, rf_id):  # 센서 관측 모델, x_true: 실제, xd: 데드 레커닝
#     x_true = motion_model(x_true, u) # 진짜 상태 업데이트

#     # add noise to gps x-y
#     z = np.zeros((0, 3)) # 노이즈가추가된거리 ,랜드마크의x ,랜드마크의y로 만들어질거임

#     for i in range(len(rf_id[:, 0])): # rf_id는 미리 만들어져있는가?

#         dx = x_true[0, 0] - rf_id[i, 0] # 실제 x값이랑 랜드마크로부터 측정된(부정확한) x값 차이
#         dy = x_true[1, 0] - rf_id[i, 1] # y값 차이
#         d = math.hypot(dx, dy) # 무슨 계싼????
#         if d <= MAX_RANGE: # 아까 정의한 최대인식거리 내부에 있는 
#             dn = d + np.random.randn() * Q_sim[0, 0] ** 0.5  # add noise 측정 노이즈
#             zi = np.array([[dn, rf_id[i, 0], rf_id[i, 1]]]) # 이걸 더해서
#             z = np.vstack((z, zi)) # # 노이즈가추가된거리 ,랜드마크의x ,랜드마크의y 이게 됨

#     # add noise to input
#     ud1 = u[0, 0] + np.random.randn() * R_sim[0, 0] ** 0.5 # 입력 노이즈 추가
#     ud2 = u[1, 0] + np.random.randn() * R_sim[1, 1] ** 0.5
#     ud = np.array([[ud1, ud2]]).T # x,y입력에 대해서 노이즈 추가한 값으로 업데이트

#     xd = motion_model(xd, ud)

#     return x_true, z, xd, ud

def gps_observation(x_true, xd, u):
    # x_true: [x, y, yaw, v]
    # xd: 데드레커닝
    # u: 인풋
    
    x_true = motion_model(x_true, u)
    
    x_gps = x_true[0,0] + np.random.randn() * math.sqrt(Q_sim[0,0])
    y_gps = x_true[1,0] + np.random.randn() * math.sqrt(Q_sim[1,1])
    
    z = np.array([[x_gps],[y_gps]])
    
    ud1 = u[0,0] + np.random.rand() * math.sqrt(R_sim[0,0])
    ud2 = u[1,0] + np.random.rand() * math.sqrt(R_sim[1,1])
    ud = np.array([[ud1],[ud2]])
    xd = motion_model(xd,ud)
    
    return x_true, z, xd, ud




def motion_model(x, u):    # 인풋에 따른 움직임 구현 x = Fx + Bu  상태벡터는 보통 (x,y,yaw,v)로 구성(differential monocycle?) 
    F = np.array([[1.0, 0, 0, 0],  # F와 상태벡터 (x,y,yaw,v)를 곱한 결과를 생각하면, x,y,yaw는 1배 곱해져서 Bu와 더해져 x를 만들고, v는 영향없음
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0], #????
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT],
                  [1.0, 0.0]])

    x = F.dot(x) + B.dot(u)

    return x


# def gauss_likelihood(x, sigma): # likelihood 계산
#     p = 1.0 / math.sqrt(2.0 * math.pi * sigma ** 2) * \
#         math.exp(-x ** 2 / (2 * sigma ** 2))

#     return p

def gauss_likelihood_my(dz,sigma):
    # dz.shape: (2,1) -> [dx,dy].T
    # sigma : 2x2
    # p(z|x) = 1 / (2*pi*sqrt(det(sigma))) * exp(-0.5 * dz^T * sigma^-1 * dz)
    det_sigma = np.linalg.det(sigma)
    inv_sigma = np.linalg.inv(sigma)
    n=2
    
    num = np.exp(-0.5 * dz.T @ inv_sigma @ dz)
    den = 2* math.pi * math.sqrt(det_sigma)
    
    return (num/den).flatten()[0]
    



def calc_covariance(x_est, px, pw): # 추정된 상태 x_est와 파티클의 위치와 가중치???? 를 이용해 공분산 계산
    """
    calculate covariance matrix
    see ipynb doc
    """
    cov = np.zeros((4, 4))
    n_particle = px.shape[1]
    for i in range(n_particle):
        dx = (px[:, i:i + 1] - x_est)
        cov += pw[0, i] * dx @ dx.T
    cov *= 1.0 / (1.0 - pw @ pw.T)

    return cov


def pf_localization(px, pw, z, u):
    """
    Localization with Particle filter
    """

    for ip in range(NP):
        #x = np.array([px[:, ip]]).T
        x = px[:,ip:ip+1]
        w = pw[0, ip]

        #  Predict with random input sampling
        ud1 = u[0, 0] + np.random.randn() * R[0, 0] ** 0.5
        ud2 = u[1, 0] + np.random.randn() * R[1, 1] ** 0.5
        ud = np.array([[ud1, ud2]]).T # 입력에 무작위 노이즈를 끼워서 노이즈가 낀 입력 만듦
        x = motion_model(x, ud)

        #  Calc Importance Weight
        # for i in range(len(z[:, 0])):
        #     dx = x[0, 0] - z[i, 1]
        #     dy = x[1, 0] - z[i, 2]
        #     pre_z = math.hypot(dx, dy)
        #     dz = pre_z - z[i, 0]
        #     w = w * gauss_likelihood_my(dz, math.sqrt(Q[0, 0]))

        dx = x[0, 0] - z[0, 0]
        dy = x[1, 0] - z[1, 0]          
        dz = np.array([[dx],[dy]])
        
        w *= gauss_likelihood_my(dz,Q)
        

        px[:, ip] = x[:, 0]
        pw[0, ip] = w

    pw = pw / pw.sum()  # normalize

    x_est = px.dot(pw.T)
    p_est = calc_covariance(x_est, px, pw)

    N_eff = 1.0 / (pw.dot(pw.T))[0, 0]  # Effective particle number
    if N_eff < NTh:
        px, pw = re_sampling(px, pw)
    return x_est, p_est, px, pw


def re_sampling(px, pw):
    """
    low variance re-sampling
    """

    w_cum = np.cumsum(pw)
    base = np.arange(0.0, 1.0, 1 / NP)
    re_sample_id = base + np.random.uniform(0, 1 / NP)
    indexes = []
    ind = 0
    for ip in range(NP): # 지정된 파티클 개수 NP 만큼, 
        while re_sample_id[ip] > w_cum[ind]:
            ind += 1
        indexes.append(ind)

    px = px[:, indexes]
    pw = np.zeros((1, NP)) + 1.0 / NP  # init weight

    return px, pw


def plot_covariance_ellipse(x_est, p_est):  # pragma: no cover
    p_xy = p_est[0:2, 0:2]
    eig_val, eig_vec = np.linalg.eig(p_xy)

    if eig_val[0] >= eig_val[1]:
        big_ind = 0
        small_ind = 1
    else:
        big_ind = 1
        small_ind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)

    # eig_val[big_ind] or eiq_val[small_ind] were occasionally negative
    # numbers extremely close to 0 (~10^-20), catch these cases and set the
    # respective variable to 0
    try:
        a = math.sqrt(eig_val[big_ind])
    except ValueError:
        a = 0

    try:
        b = math.sqrt(eig_val[small_ind])
    except ValueError:
        b = 0

    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eig_vec[1, big_ind], eig_vec[0, big_ind])
    fx = rot_mat_2d(angle) @ np.array([[x, y]])
    px = np.array(fx[:, 0] + x_est[0, 0]).flatten()
    py = np.array(fx[:, 1] + x_est[1, 0]).flatten()
    plt.plot(px, py, "--r")


def main():
    print(__file__ + " start!!")

    time = 0.0

    # RF_ID positions [x, y]
    rf_id = np.array([[10.0, 0.0],
                      [10.0, 10.0],
                      [0.0, 15.0],
                      [-5.0, 20.0]])

    # State Vector [x y yaw v]'
    x_est = np.zeros((4, 1))
    x_true = np.zeros((4, 1))

    px = np.zeros((4, NP))  # Particle store
    pw = np.zeros((1, NP)) + 1.0 / NP  # Particle weight
    x_dr = np.zeros((4, 1))  # Dead reckoning

    # history
    h_x_est = x_est
    h_x_true = x_true
    h_x_dr = x_true

    while SIM_TIME >= time:
        time += DT
        u = calc_input()

        # x_true, z, x_dr, ud = gps_observation(x_true, x_dr, u, rf_id)
        x_true, z, x_dr, ud = gps_observation(x_true, x_dr, u)
        x_est, PEst, px, pw = pf_localization(px, pw, z, ud)

        # store data history
        h_x_est = np.hstack((h_x_est, x_est))
        h_x_dr = np.hstack((h_x_dr, x_dr))
        h_x_true = np.hstack((h_x_true, x_true))

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

            # for i in range(len(z[:, 0])):
            #     plt.plot([x_true[0, 0], z[i, 1]], [x_true[1, 0], z[i, 2]], "-k")
            #plt.plot(rf_id[:, 0], rf_id[:, 1], "*k")
            plt.plot([x_true[0, 0], z[0, 0]], [x_true[1, 0], z[1, 0]], "-k", label="Measurement line")
            plt.plot(px[0, :], px[1, :], ".r")
            plt.plot(np.array(h_x_true[0, :]).flatten(),
                     np.array(h_x_true[1, :]).flatten(), "-b")
            plt.plot(np.array(h_x_dr[0, :]).flatten(),
                     np.array(h_x_dr[1, :]).flatten(), "-k")
            plt.plot(np.array(h_x_est[0, :]).flatten(),
                     np.array(h_x_est[1, :]).flatten(), "-r")
            plot_covariance_ellipse(x_est, PEst)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)


if __name__ == '__main__':
    main()