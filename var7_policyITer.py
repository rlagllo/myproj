import numpy as np

def get_state(state,action): 
    action_perform = [(-1,0),(1,0),(0,-1),(0,1)] # up down left right
    state[0] += action_perform[action][0] # state[0]: 행 데이터 -> y좌표
    state[1] += action_perform[action][1] # state[1]: 열 데이터 -> x좌표

    if state[0]<0:
        state[0] = 0
    elif state[0]>3:
        state[0] = 3
    if state[1]<0:
        state[1] = 0
    elif state[1]>3:
        state[1] = 3
    return state[0], state[1]
    

def policy_eval(grid_height, grid_width, action, policy, iter_num,reward=-1,dis=1):
    post_value_table = np.zeros([grid_height,grid_width],dtype=float)
    if iter_num == 0:
        print('Iteration: {}\n{}\n'.format(iter_num,post_value_table))
        return post_value_table
    
    for iteration in range(iter_num):   
        next_value_table = np.zeros([grid_height,grid_width],dtype=float)
        for i in range(grid_height):
            for j in range(grid_width):
                value_t = 0
                if i == j and ((i==0) or (i == 3)):
                    value_t = 0
                else:
                    for act in action:
                        pi,pj = get_state([i,j],act)
                        value = policy[i][j][act] * (reward + dis*post_value_table[pi][pj]) # <----------------------처음에 주어진 policy 가지고 지침(가치함수)을 만드는 부분
                        value_t += value # 각 행동을 구분하지 않고 모든 행동의 가치를 합치면 q 함수가 아니고 가치함수라고 부름
                next_value_table[i][j] = round(value_t,3)
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
    return next_value_table # 최신 가치 테이블: post_value_table, 곧 업데이트 될 최최신 가치 테이블: next_value_table임. next post 뭘 반환하든 같음 가장 마지막 순간의 것을 반환하기 때문
    

def policy_improve(value, action, policy, grid_width):
    grid_height = grid_width
    action_label = ['up', 'down', 'left', 'right']
    action_table = []
    for i in range(grid_height):
        for j in range(grid_width):
                q_func_list = [] # q 함수는 
                if i==j and (i == 0 or i == 3):
                    action_table.append('T')
                else:
                    for k in action: # for k in action이라 해도 됨
                        pi,pj = get_state([i,j],k)
                        q_func_list.append(value[pi][pj])
                    max_actions = [action_v for action_v,x in enumerate(q_func_list) if x == max(q_func_list)] # 확률 최대인 애들의 확률은 x, 걔들이 몇 번째인지는 action_v
                    policy[i][j] = [0]*len(action) # policy[i][j][k]인데 i,j위치에서 할 수 있는 행동들의 확률을 0으로 만든 후에, max q 인 애들한테만 확률을 주려고, 초기화
                    for y in max_actions:
                        policy[i][j][y] = 1/len(max_actions)
                    idx = np.argmax(policy[i][j]) # 위에서 i,j에서의 policy값을 max q인 action들한테만 확률을 배분하는 걸로 바꾼 후에, 확률을 얻은 애들의 인덱스
                    action_table.append(action_label[idx]) # action_table은 수정된 policy의 내용을 action_label을 써서 up doww 등으로 나타낼거임
                    # policy는 그럼 각 위치에서, '이 위치에서는 이 행동들을 선택하라'는 내용을 담고있음. 확률로써
    action_table = np.asarray(action_table).reshape((grid_height,grid_width))
    # grid_height, grid_width만큼의 좌표에서, 각 좌표에 해당하는 policy가 있는데, 그 모든 좌표가 1차원으로 된게 action_table인데, 이걸 다시 2차원 좌표의 형태로 만들려는 거임
    print('Updated policy is :\n{}\n'.format(policy))
    print('at each state, chosen action is :\n{}'.format(action_table))
  
    return policy
    

grid_width = 4
grid_height = grid_width
action = [0,1,2,3]
policy = np.empty([grid_height,grid_width,len(action)],dtype=float) # H,W,C 쓰레기값으로 생성, 바로 밑 for문에서 알맞게 설정
# policy: i,j,k=1,2,2라면, 1,2에서, '2'선택지를 선택할 확률이 0.25임. 이런식으로 모든 좌표에서 각 선택지를 선택할 확률을 정해놓은게 policy
# 정책 초기화 같은 느낌 모든 선택지가 25%로 나눠가짐
for i in range(grid_height):
    for j in range(grid_width):
        for k in action:
            if i == j and ((i==0) or (i==3)):
                policy[i][j][k] = 0.00
            else:
                policy[i][j][k] = 0.25

value = policy_eval(grid_height, grid_width, action, policy, 100) # 현재 policy를 기반으로, 각 상태(위치)의 가치를 매긴 결과
updated_policy = policy_improve(value, action, policy,grid_width)
# 현재 policy로 각 상태를 값을 매겼고, 이제부터는 


#post_value_table은 현재까지의 정책 평가 결과로, 각 상태(위치)에서의 미래의 가치를 담고 있습니다.

# 처음에는 모든 위치에서의 가치가 0으로 초기화되어 있습니다. 그리고 첫 번째 시행(첫 번째 iteration)에서 policy_evaluation 함수가 실행되면, 그때부터 각 위치에서 가능한 행동을 바탕으로 미래 가치를 계산하기 시작합니다. 이 과정에서 post_value_table은 현재까지 계산된 가치를 담게 되며, 이전 단계에서의 가치를 반영하여 업데이트됩니다.

# 구체적으로:

# 첫 번째 시행에서는 post_value_table이 0으로 시작하고, 각 위치에서 가능한 행동에 대해 그 가치를 계산하면서 업데이트됩니다. 그 후, 각 위치에서의 가치는 reward와 dis * post_value_table[pi][pj]를 기반으로 계산됩니다. 이때 post_value_table[pi][pj]는 이전 위치의 가치를 의미합니다.

# 이후 시행에서는 post_value_table이 이전 평가 결과를 바탕으로 계속 업데이트됩니다. 즉, post_value_table은 각 상태에서의 미래 가치를 점진적으로 반영하면서, 계속 개선되고, 시간이 지날수록 각 상태에서의 정확한 가치가 더 잘 반영되게 됩니다.

# 그래서 post_value_table은 과거의 평가로 생성된 미래의 가치라 할 수 있습니다. 이 값이 점진적으로 바뀌면서 최적의 정책을 찾기 위한 계산이 이루어지는 것입니다.