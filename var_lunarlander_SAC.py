import gymnasium as gym

import os
from stable_baselines3 import SAC
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# 환경 로드
env = gym.make("LunarLanderContinuous-v3")

log_dir = "C:\\Users\\김해창\\Desktop\\lunarlander"
os.makedirs(log_dir, exist_ok=True)

# SAC 모델 학습
model = SAC("MlpPolicy", env, verbose=1, device='cuda')
model.learn(total_timesteps=100000)
# 모델 저장
model.save(log_dir + "\\sac_lunarlander_final")
# 학습된 모델 테스트
# obs = env.reset()
# for _ in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#         obs, _ = env.reset()



env.close()

# 환경 생성
env = gym.make("LunarLanderContinuous-v2")

# 학습된 모델 로드
model = SAC.load(log_dir + "\\sac_lunarlander_final", device='cuda')

# 환경 실행 및 랜더링
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)  # 결정론적 행동 선택
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs, _ = env.reset()

env.close()