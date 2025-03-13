import numpy as np

def get_state(state, action): # state: 현재위치, action: 입력된 행동
    
  action_grid = [(-1, 0), (1, 0), (0, -1), (0, 1)]
  
  state[0]+=action_grid[action][0]
  state[1]+=action_grid[action][1]
  
  if state[0] < 0:
    state[0] = 0
  elif state[0] > 3: # 0,0 ~ 3,3 범위 사각형 내부에서만 움직임
    state[0] = 3
  
  if state[1] < 0:
    state[1] = 0
  elif state[1] > 3:
    state[1] = 3
  
  return state[0], state[1] # action 하나 적용 후 다음 위치 반환

# 여기서 action은 [0,1,2,3]: 상하좌우
def policy_evaluation(grid_width, grid_height, action, policy, iter_num, reward=-1, dis=1):
    
  # table initialize
  post_value_table = np.zeros([grid_height, grid_width], dtype=float)
  
  # iteration
  if iter_num == 0:
    print('Iteration: {} \n{}\n'.format(iter_num, post_value_table))
    return post_value_table
  
  for iteration in range(iter_num):
    next_value_table = np.zeros([grid_height, grid_width], dtype=float)
    for i in range(grid_height):
      for j in range(grid_width):
        if i == j and ((i == 0) or (i == 3)):
          value_t = 0
        else :
          value_t = 0
          for act in action:
            # i,j는 지금 위치, i_,j_는 action의 한 act를 수행했을 때 결과위치
            # policy에 각 state에서 action을 취할 확률을 저장함. 
            # policy[i][j][act]: 확률, reward, dis, 수행 전 위치 post_value_table을 조합해서 value(확률) 계산
            i_, j_ = get_state([i,j], act)
            value = policy[i][j][act] * (reward + dis*post_value_table[i_][j_]) # <----------------------처음에 주어진 policy 가지고 지침(가치함수)을 만드는 부분
            # policy[i][j][k] = 0.25라서, 0,1에서 '위'를 취할 확률은 0.25임(policy[0][1][0])
            # reward는 모두 공평하게 -1이고, dis는 reward의 반영정도, post_value_table
            value_t += value # 한 위치에서 할 수 있는 모든 선택(상하좌우)의 가치를 전부 더한 것
            # 만약 모든 가치를 합산한 post_value_table이 있고, 그 가치가 높을 때
            # 현재시점에서는, 그 쪽 위치로 이동하는 편이 전체적인 가치가 올라가니까 그쪽으로 가려고 할 것임
        next_value_table[i][j] = round(value_t, 3)
    # 그럼 하나의 iteration에서, 모든 i,j좌표에 대해 하나의 post_value_table(처음 iteration때는 전부 0인 그리드)로 수행
    # 그걸 post_value_table에 한 번 저장
    # 이제 다음 iteration 동안에는 그 post_value_table로 value를 계산

    # print result
    if (iteration % 10) != iter_num: 
      # print result 
      if iteration ==1 or (iteration ==2): 
        print('Iteration: {} \n{}\n'.format(iteration, next_value_table))
      if iteration > 100 :
        if (iteration % 20) == 0: 
          print('Iteration: {} \n{}\n'.format(iteration, next_value_table))
      else :
        if (iteration % 10) == 0:
          print('Iteration: {} \n{}\n'.format(iteration, next_value_table))
    else :
      print('Iteration: {} \n{}\n'.format(iteration, next_value_table ))
    
    post_value_table = next_value_table
      
  return next_value_table

grid_width = 4
grid_height = grid_width
action = [0, 1, 2, 3] # up, down, left, right
policy = np.empty([grid_height, grid_width, len(action)], dtype=float)
for i in range(grid_height):
  for j in range(grid_width):
    for k in action:
      if i==j and ((i==0) or (i==3)):
        policy[i][j][k]=0.00
      else :
        policy[i][j][k]=0.25

value = policy_evaluation(grid_width, grid_height, action, policy, 100)
# 지금까지는 아무 편향도 없고 0,0 & 3,3에서만 어떤 행동도 하지 않게 설정한 policy로 각 지점에서의 행동 확률을 계산해본거임


def policy_improvement(value, action, policy, reward = -1, grid_width = 4):
    
  grid_height = grid_width
  
  action_match = ['Up', 'Down', 'Left', 'Right']
  action_table = []
  
  # get Q-func.
  for i in range(grid_height):
    for j in range(grid_width):
      q_func_list=[]
      if i==j and ((i==0)or (i==3)): # 0,0 3,3일 경우에는 action_table에 'T'넣고 바로 패스
        action_table.append('T')
      else:
        for k in action: # for k in action이라 해도 됨
          i_, j_ = get_state([i, j], k)
          q_func_list.append(value[i_][j_]) # 전반부에 아무 편향도 없는 상태로 만든 value table 이용
        max_actions = [action_v for action_v, x in enumerate(q_func_list) if x == max(q_func_list)] # []안에 정의했으니까 리스트임
        # q_func_list 중에서 action_v,x = 인덱스, 원소의 값 중 최대인 x에 해당하는 action_v를 max_actions에 저장
        # 하나의 i,j 좌표에서, q-list에는 상하좌우 4가지 선택지에 대한 value가 있게 됨
        # 그 i,j좌표에서는 value가 가장 큰 값을 선택하는 것이고, 그것은 그 좌표에서 어디로 이동할 지를 정하는 것

        # update policy 여기서는 원래 있던 policy값 [i][j]에서의 k 값들, 여기서는 len(action)=4니까 0,1,2,3(상하좌우) 얘들을 0으로 만들고,
        # 바로 위에서 계산한 대로 가장 큰, 미래 기대 가치가 높은 선택지들에게만 균등하게 확률을 나눠줌
        policy[i][j]= [0]*len(action) # policy[i][j][k]인데 i,j위치에서 할 수 있는 행동들의 확률을 0으로 만든 후에, max q 인 애들한테만 확률을 주려고, 초기화
        for y in max_actions :
          policy[i][j][y] = (1 / len(max_actions))

        # get action
        # action_table은 현재 상태를 가지고 만들어져서, 바로 다음번에 유효한, 각 좌표에서 선택해야 할 다음 행동의 맵
        idx = np.argmax(policy[i][j]) # i,j에서는 무슨 행동을 가장 해야 하는가? idx = 2라면, action[2]: left를 선택해야한다.
        action_table.append(action_match[idx])
  action_table=np.asarray(action_table).reshape((grid_height, grid_width))                
  
  print('Updated policy is :\n{}\n'.format(policy))
  print('at each state, chosen action is :\n{}'.format(action_table))
  
  return policy

updated_policy = policy_improvement(value, action, policy)

# 이 코드는 0,0 & 3,3에 다가가는 방향으로 정책을 수정하는 코드임

# Iteration: 1 
# [[ 0. -1. -1. -1.]
#  [-1. -1. -1. -1.]
#  [-1. -1. -1. -1.]
#  [-1. -1. -1.  0.]]

# Iteration: 2
# [[ 0.   -1.75 -2.   -2.  ]
#  [-1.75 -2.   -2.   -2.  ]
#  [-2.   -2.   -2.   -1.75]
#  [-2.   -2.   -1.75  0.  ]]

# 아그럼 iteration 2를 할 때, 0,1의 경우 왼쪽'을 선택했을때 -> post_value_table[i_][j_]값은 0
# '오른쪽'일 때는 -1이니까 value계산이 정중앙에 있는 위치와 다르게 나옴

#현재에 안 좋은 선택을 해야 결과적으로 더 나은 방향으로 가는 경우도 분명히 있을 수 있음 -> 탐색(exploration)과 착취(exploitation)