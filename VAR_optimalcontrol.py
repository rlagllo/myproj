import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are
from scipy.integrate import solve_ivp
from scipy.signal import cont2discrete
import matplotlib.pyplot as plt
from numpy.linalg import solve, norm


# system matrices
A = np.array([       # state 행렬, 시스템의 내재된 동작 (위치 속도 가속도)
    [ 0.0,  0.0,  1.0,  0.0],
    [ 0.0,  0.0,  0.0,  1.0],
    [-3.0,  2.0, -2.0,  1.0],
    [ 1.0, -1.0,  0.5, -1.0]
])
B = np.array([       # input 행렬 4x2
    [0.0, 0.0],
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 0.5]
])
n = A.shape[0]
m = B.shape[1]
print("B.shape[1] = ", B.shape[1])
# continuous-time lqr
Q = np.diag([1e2, 1e2, 1e0, 1e0])    # state-error에 대한 가중치
R = 1e-2*np.eye(2)                   # 제어 입력에 대한 가중치


max_iters = 1000
tolerance = 1e-9
K = np.zeros((m,n))
for _ in range(max_iters):
    A_BK = A-B@K
    l = np.eye(n)
    X = np.kron(l,A_BK.T) + np.kron(A_BK.T,l)
    vecQ_KRK = (Q + K.T @ R @ K).flatten()
    
    vecP = solve(-X, vecQ_KRK)
    P = vecP.reshape((n,n))
    K_new = np.linalg.inv(R) @ B.T @P
    if norm(K_new- K) < tolerance:
        K = K_new
        break
    K = K_new
    
P_lib = solve_continuous_are(A, B, Q, R)
K_lib = np.linalg.inv(R) @ B.T @ P

P = solve_continuous_are(A, B, Q, R) # ARE: algebra ricatti eqn 리카티 방정식 풀고 P 행렬 계산
K = np.linalg.inv(R) @ B.T @ P       # 최적 제어 gain K 계산 K= R^-1 * B_T * P로 계산해서 x_ref - x 에 대한 피드백을 통해 control input 결정

# initail state and reference
x0 = np.array([0.0, 0.0, 0.0, 0.0])
xref = np.array([0.5, 1.0, 0.0, 0.0])

# simulation  K_lib & P_lib
sol_lib = solve_ivp(
    lambda t, x: A @ x + B @ K_lib @ (xref - x),  # closed loop system
    (0.0, 5.0),  # simulation time interval
    x0,  # initial state
    t_eval=np.linspace(0.0, 5.0, 501)
)

# simulation  K&P
sol = solve_ivp(
    lambda t, x: A @ x + B @ K @ (xref - x),  # closed loop system
    (0.0, 5.0),  # simulation time interval
    x0,  # initial state
    t_eval=np.linspace(0.0, 5.0, 501)
)








# discrete-time system
Ts = 0.05  # sampling time
Ad, Bd, _, _, _ = cont2discrete((A, B, np.eye(n), np.zeros_like(B)), Ts)

Pd = Q.copy()

for _ in range(max_iters):
    Pd_next = Ad.T @ Pd @ Ad + Q - Ad.T @ Pd @ Bd @ np.linalg.inv(R + Bd.T @ Pd @ Bd) @ Bd.T @ Pd @ Ad
    if norm(Pd_next - Pd) < tolerance:
        Pd = Pd_next
        break
    Pd = Pd_next
Pd_lib = solve_discrete_are(Ad,Bd,Q,R)
Kd_lib = np.linalg.inv(R+Bd.T @ Pd @ Bd) @ Bd.T @ Pd @ Ad

    
    
    
    
Kd = np.linalg.inv(R + Bd.T @ Pd @ Bd) @ Bd.T @ Pd @ Ad


# simulation
t = np.arange(0.0, 5.0, Ts)  # simulation time sequence
y = [x0.copy()]  # list of system state
for _ in t[1:]:
    y.append(Ad @ y[-1] + Bd @ Kd @ (xref - y[-1]))
y = np.array(y).T

y_lib = [x0.copy()]  # List of system state
for _ in t[1:]:
    y_lib.append(Ad @ y_lib[-1] + Bd @ Kd_lib @ (xref - y_lib[-1]))
y_lib = np.array(y_lib).T

# figure plot
plt.figure("continuous-time system")
for k, label in enumerate(["p1", "p2", "v1", "v2"]):
    plt.subplot(411+k)
    plt.plot(sol.t, xref[k] * np.ones_like(sol.t), "k--")
    plt.plot(sol.t, sol.y[k, :])
    plt.plot(sol_lib.t, sol_lib.y[k, :], "b--", label="lib Solution")
    plt.legend()
    plt.ylabel(label)
plt.xlabel("t")
plt.figure("discrete-time system")
for k, label in enumerate(["p1", "p2", "v1", "v2"]):
    plt.subplot(411+k)
    plt.plot(t, xref[k] * np.ones_like(t), "k--")
    plt.plot(t, y[k, :])
    plt.plot(t, y_lib[k, :], "b--", label="lib Solution")
    plt.legend()
    plt.ylabel(label)
plt.xlabel("t")
plt.show()