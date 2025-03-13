import gym
import numpy as np
""" 공식 홈피 내용
observation space : Discrete(16) 
starting state: episode starts with the player in state[0] (location [0,0])
The observation is a value representing current position as current_row * ncols + current_col
(where both the row and col start at 0).
For example, the goal position in the 4x4 map can be calculated as follows: 3 * 4 + 3 = 15.
S  1  2  3
4  5  6  7
8  9 10 11
12 13 14  G에서 goad은 15임
The number of possible observations is dependent on the size of the map.
The observation is returned as an int().

step() and reset() return a dict with the following keys: 상태전이 확률
p - transition probability for the state

Reward:
goal +1
hole/frozen +0

Termination:
The player moves into a hole.
The player reaches the goal at max(nrow) * max(ncol) - 1 (location [max(nrow)-1, max(ncol)-1]).
Truncation (when using the time_limit wrapper):
The length of the episode is 100 for 4x4 environment, 200 for FrozenLake8x8-v1 environment.

action space : Discrete(4)
가능한 행동 -> 0,1,2,3    left down right up
"""
desc = [
    "FFFFHFFF",
    "FFFFFFFH",
    "FFFHFFFF",
    "FFFFFHFH",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "GFHFFFHS",
]
slippery = False
env = gym.make('FrozenLake-v1', is_slippery = slippery, map_name = '8x8',render_mode = 'ansi',desc = desc)
env.reset()
states_size = env.observation_space.n # discrete한 속성은 .n 붙음
action_size = env.action_space.n
print(states_size) # 64 map size
print(action_size) # 4
policy = np.ones([states_size,action_size],dtype=float)

for state in range(states_size): # 정책 초기화
    row, col = state//8, state % 8
    if row == 0:
        policy[state,3] = 0
    if row == 7:
        policy[state,1] = 0
    if col == 0:
        policy[state,0] = 0
    if col == 7:
        policy[state,2] = 0
    total_prob = np.sum(policy[state])
    if total_prob > 0:
        policy[state] /= total_prob # 확률합이 1이 되도록 정규화
for state in range(states_size):
    print(f"{state} state: {policy[state]}")
gamma = 1
theta = 1e-8
# frozenlake는 확률함수 P랑 보상 R을 기본적으로 제공해줌
"""
P(s'| s,a): 상태전이확률, s에서 액션 a를 취할때, s'상태가 될 확률
내 policy는 state 5 위치에서 0.25의 확률로 action 3(right)을 취할건데
이때, 실제로 s' = state 6으로 이동할 확률은 P(6|5,3) = frozenlake에서 이미 설정된 값임
"""
# policy_eval 함수 흐름: 처음에 초기화한 policy(행동확률)를 가지고, 모든 배열 값이 0인 배열(value)에, bellman eqn 사용해서 계산한 값 넣기
def policy_eval(policy, states_size, action_size, gamma, theta): # gamma: discount factor, theta: threshold
    V = np.zeros([states_size]) # value table 초기화
    count = 1
    while True:
        delta = 0
        newv = np.copy(V) # 얘도 사이즈16짜리 1차원 배열 0으로 초기화
        for state in range(states_size):
            v = 0
            for a,action_prob in enumerate(policy[state]): # 해당 state에서, 4개의 액션들
                # env.P[state][a]는, state에서, a를 취했을 때
                # p의 확률로, sprime 상태로 전이되고, 그때 받을 reward, 그때가 done인지 아닌지  를 반환
                # 하나의 state에서, 가능한 action이 3개라면, P[state][action]은 총 3개의 세트를 반환할거임
                for p,sprime,reward,done in env.P[state][a]:
                    v += action_prob * p * (reward + gamma*V[sprime])
            newv[state] = v
            delta = max(delta,abs(V[state] - newv[state]))
        V = newv
        if delta<theta:
            print("final count:",count)
            break
        count += 1
        if count % 10 == 0 or count == 8:
            print(f"count: {count}\n{V.reshape(8,8)}\n")
    return V # threshold 지점까지 반복해서 업뎃 후 value table 반환
x = policy_eval(policy, states_size, action_size, gamma, theta) # 현재 정책으로 산출된 상태가치
print(env.render())

def policy_improve(x, policy, states_size, action_size, gamma, env):
    new_policy = np.zeros(states_size)
    for state in range(states_size):
        conven_action = np.argmax(policy[state]) # 기존 정책을 따를 때 state에서 취해야 할 행동
        Qval = np.zeros(action_size) # 한 state에 대해 
        for act in range(action_size):
            Qval[act] = 0
            for transition in env.P[state][act]:
                p, sprime, reward, done = transition
                Qval[act] += p * (reward + gamma * x[sprime])
        best_choice = np.argmax(Qval) # 해야하는 행동의 번호
        new_policy[state]= best_choice
    return new_policy
env.close()
new_policy = policy_improve(x, policy, states_size, action_size, gamma, env)
print(new_policy.reshape(8,8))
env = gym.make('FrozenLake-v1', is_slippery = slippery, map_name = '8x8',render_mode = 'human',desc = desc)
state = env.reset()[0]
done = False

while not done:
    action = int(new_policy[state])
    state,reward,done,_,info = env.step(action)
    env.render()
    
env.close()

