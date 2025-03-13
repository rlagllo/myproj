import matplotlib.pyplot as plt
import control as ctrl

# 분자와 분모 계수 정의
numerator = [100, 100 * (0.4 + 0.224), 100 * 0.4 * 0.224]  # (s + 0.4)(s + 0.224)
denominator = [1, (1 + 5 + 7.56 + 0.011), (5 * 7.56 + 7.56 * 0.011 + 5 * 0.011 + 1 * 7.56), 
               (5 * 7.56 * 0.011 + 1 * 5 * 7.56 + 1 * 0.011 * 7.56 + 1 * 5 * 0.011), 5 * 7.56 * 0.011, 0]  # s(s+1)(s+5)(s+7.56)(s+0.011)

# 전달 함수 정의
G = ctrl.TransferFunction(numerator, denominator)

# Bode Diagram 그리기
mag, phase, omega = ctrl.bode(G, dB=True, Hz=False, deg=True, plot=True)

# 그래프 표시
plt.show()
