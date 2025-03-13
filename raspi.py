import lgpio as GPIO
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # 백엔드를 TkAgg로 설정
import matplotlib.pyplot as plt
import cv2
from picamera2 import Picamera2, Preview
import time
import lgpio
from PIL import Image
import skimage.transform


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, down_sample=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride

        if down_sample:
            self.down_sample = IdentityPadding(in_channels, out_channels, stride)
        else:
            self.down_sample = None

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            shortcut = self.down_sample(x)

        out += shortcut
        out = self.relu(out)
        return out


class IdentityPadding(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(IdentityPadding, self).__init__()

        self.pooling = nn.MaxPool2d(1, stride=stride)
        self.add_channels = out_channels - in_channels

    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.add_channels))
        out = self.pooling(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_layers, block, num_classes=10):
        super(ResNet, self).__init__()
        self.num_layers = num_layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # feature map size = 32x32x16
        self.layers_2n = self.get_layers(block, 16, 16, stride=1)
        # feature map size = 16x16x32
        self.layers_4n = self.get_layers(block, 16, 32, stride=2)
        # feature map size = 8x8x64
        self.layers_6n = self.get_layers(block, 32, 64, stride=2)

        # output layers
        self.avg_pool = nn.AvgPool2d(8, stride=1)
        self.fc_out = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_layers(self, block, in_channels, out_channels, stride):
        if stride == 2:
            down_sample = True
        else:
            down_sample = False

        layers_list = nn.ModuleList(
            [block(in_channels, out_channels, stride, down_sample)]) 

        for _ in range(self.num_layers - 1):
            layers_list.append(block(out_channels, out_channels))

        return nn.Sequential(*layers_list)

    def forward(self, x):
        x = self.conv1(x) # 64, 3, 32, 32 -> 64, 16, 32, 32
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layers_2n(x) # 64, 16, 32, 32
        x = self.layers_4n(x) # 64, 32, 16, 16
        x = self.layers_6n(x) # 64, 64, 8, 8

        feature = x.clone()
        x = self.avg_pool(x)
        #print(f"x.shape: {x.shape}") # 64, 64, 1, 1
        
        x = x.view(x.size(0), -1)
        x = self.fc_out(x)
        return x, feature



def resnet():
    block = ResidualBlock
    model = ResNet(5, block)
    return model

transforms_test = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])
model_path = "/home/pikim/.vscode/myenv/model_resnet.pth"
net = resnet()

state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)

net.load_state_dict(state_dict, strict=True)

net.eval()

# Picamera2 ��ü ����
picam2 = Picamera2()

# �̸�����(Preview) ���� ����
config = picam2.create_preview_configuration()
picam2.configure(config)


# TRIG = 23
# ECHO = 24
# PWM_GPIO = 12 
h = GPIO.gpiochip_open(0)
pins = [14, 15, 18, 23]
for pin in pins:
    lgpio.gpio_claim_output(h, pin)
    
step_sequence = [
    (1, 0, 0, 0),
    (1, 1, 0, 0),
    (0, 1, 0, 0),
    (0, 1, 1, 0),
    (0, 0, 1, 0),
    (0, 0, 1, 1),
    (0, 0, 0, 1),
    (1, 0, 0, 1)
]

delay = 0.0008
total_steps = 1000
def move_motor(total_steps):
    for i in range(total_steps):
        for step in step_sequence:
            lgpio.gpio_write(h, pins[0], step[0])
            lgpio.gpio_write(h, pins[1], step[1])
            lgpio.gpio_write(h, pins[2], step[2])
            lgpio.gpio_write(h, pins[3], step[3])
            time.sleep(delay)
        for pin in pins:
            lgpio.gpio_write(h, pin, 0)
# GPIO.gpio_claim_output(h,TRIG)
# GPIO.gpio_claim_output(h,PWM_GPIO)
# GPIO.gpio_claim_input(h,ECHO)


# def set_servo_angle(angle):
#     """����(0~180)? PWM ?????? ????????? (0~100%)?? ?????"""
#     if angle < 0 or angle > 180:
#         raise ValueError("Angle must be between 0 and 180")
    
#     # ???����?????? PWM ?????? ????????? ���� ????? (5%~10%)
#     duty_cycle = (angle / 180.0) * 5 + 5  # 0???=5%, 180???=10%           %?? ����??
#     print("����?? ???????? duty cycle: ", duty_cycle)
#     print(f"Setting angle: {angle} -> PWM Duty Cycle: {duty_cycle}%")
#     GPIO.tx_pwm(h, PWM_GPIO, 50, duty_cycle)  # 50Hz PWM ??????   
    
# def get_distance():
#     GPIO.gpio_write(h,TRIG,0)
#     time.sleep(2)
#     GPIO.gpio_write(h,TRIG,1)
#     time.sleep(0.00001)
#     GPIO.gpio_write(h,TRIG,0)
#     while GPIO.gpio_read(h,ECHO) ==0:
#         pulse_start = time.time()
#     while GPIO.gpio_read(h,ECHO) ==1:
#         pulse_end = time.time()

#     pulse_duration = pulse_end - pulse_start
#     distance = round(pulse_duration*17150, 2)
    
#     return distance

# /home/pikim/.vscode/myenv/my.py
picam2.stop()
classes =  ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

if __name__ == "__main__":
    params = list(net.parameters())[-2]
    picam2.start()
    time.sleep(2)
    try:
        while True: # cv2: uint8 HWC 0~255     plt: float&uint  HWC 0~1 or 0~255(uint)
            #picam2.start()
            image_array = picam2.capture_array() # (480, 640, 4)
            #print(image_array.shape)
            image_3ch = cv2.cvtColor(image_array, cv2.COLOR_BGRA2RGB) # RGB
            #print(image_3ch.shape)
            pil_image = Image.fromarray(image_3ch)
            image_3ch_resized = transforms_test(pil_image).unsqueeze(0)
            classification, feature = net(image_3ch_resized)
            prediction = torch.argmax(classification,1)
            if classes[int(prediction)] == 'dog':
                print("captured is dog!\n moving motor")
                #move_motor(total_steps)
                overlay = params[int(prediction)].matmul(feature.reshape(64,8*8)).reshape(8,8).cpu().data.numpy()
                overlay = overlay-np.min(overlay)
                overlay = overlay / np.max(overlay)
                overlay_resized = skimage.transform.resize(overlay, [480, 640])

                overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB) # RGB
                original = image_3ch_resized.cpu().squeeze(0).numpy() # 아직까지 토치텐서 N C H W    1 3 32 32
                print(original.shape)
                original = np.transpose(original , (1,2,0))
                original = skimage.transform.resize(original,[128,128])
                

                plt.imshow(image_3ch)
                plt.imshow(overlay_resized, alpha = 0.4, cmap = 'jet')
                plt.show()
            #cv2.imshow("Captured Image",image_array)

            else:
                print("not a dog")
            continue
            #cv2.waitKey(0)
            
            # if classes[int(prediction)] == 'dog':
            #     print("captured is dog!\n moving motor")
            #     #move_motor(total_steps)
            # else:
            #     print("not a dog")
            #picam2.stop()
            
            # dist = get_distance()
            # if dist < 10:
            #     set_servo_angle(0)
            #     time.sleep(1)
            #     set_servo_angle(170)  # 90???
            #     time.sleep(1)
                # set_servo_angle(0)
                # time.sleep(0.5)
            # print("measured distance = {:.2f}cm".format(dist))
            # time.sleep(0.1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        
    finally:
        picam2.stop()
        GPIO.gpiochip_close(h)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        import lgpio as GPIO
import time

TRIG = 23
ECHO = 24
PWM_GPIO = 12 
h = GPIO.gpiochip_open(0)
GPIO.gpio_claim_output(h,TRIG)
GPIO.gpio_claim_output(h,PWM_GPIO)
GPIO.gpio_claim_input(h,ECHO)
def set_servo_angle(angle):
    
    
    """각도(0~180)를 PWM 듀티 사이클 (0~100%)로 변환"""
    if angle < 0 or angle > 180:
        raise ValueError("Angle must be between 0 and 180")
    
    # 서보모터의 PWM 듀티 사이클 범위 변환 (5%~10%)
    duty_cycle = (angle / 180.0) * 5 + 5  # 0도=5%, 180도=10%           %로 맞추기
    print("각도로 변환된 duty cycle: ", duty_cycle)
    print(f"Setting angle: {angle} -> PWM Duty Cycle: {duty_cycle}%")
    GPIO.tx_pwm(h, PWM_GPIO, 50, duty_cycle)  # 50Hz PWM 적용   
    
    
def get_distance():
    GPIO.gpio_write(h,TRIG,0)
    time.sleep(2)
    GPIO.gpio_write(h,TRIG,1)
    time.sleep(0.00001)
    GPIO.gpio_write(h,TRIG,0)
    while GPIO.gpio_read(h,ECHO) ==0:
        pulse_start = time.time()
    while GPIO.gpio_read(h,ECHO) ==1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = round(pulse_duration*17150, 2)
    
    return distance

if __name__ == "__main__":
    try:
        while True:
            dist = get_distance()
            if dist < 10:
                set_servo_angle(0)
                time.sleep(1)
                set_servo_angle(170)  # 90도
                time.sleep(1)
                # set_servo_angle(0)
                # time.sleep(0.5)
            print("measured distance = {:.2f}cm".format(dist))
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        GPIO.giochip_close(h)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # from gpiozero import OutputDevice, PWMOutputDevice
# from time import sleep
# from gpiozero.pins.mock import MockFactory
# from gpiozero import Device
# Device.pin_factory = MockFactory()
# from gpiozero import LED
# from gpiozero.pins.pigpio import PiGPIOFactory

# factory = PiGPIOFactory()
# IN1 = OutputDevice(22)
# IN2 = OutputDevice(23)
# IN3 = OutputDevice(24)
# IN4 = OutputDevice(25)
# ENA = PWMOutputDevice(18,pin_factory=factory)
# ENB = PWMOutputDevice(27)

# step_sequence = [
#     [1,0,0,0],
#     [1,1,0,0],
#     [0,1,0,0],
#     [0,1,1,0],
#     [0,0,1,0],
#     [0,0,1,1],
#     [0,0,0,1],
#     [1,0,0,1]
# ]

# def set_step(w1,w2,w3,w4):
#     IN1.value = w1
#     IN2.value = w2
#     IN3.value = w3
#     IN4.value = w4
    

# def step_motor(steps, direction = 1, delay = 0.01):
#     for _ in range(steps):
#         for step in (step_sequence if direction > 0 else reversed(step_sequence)):
#             lines.set_values(step)
#             sleep(delay)
# try:
#     while True:
#         steps = int(input("Enter steps: "))
#         direction = int(input("Enter direction: "))
#         step_motor(steps,direction)
# except  KeyboardInterrupt:
#         print("keyboard interrupt")


# import gpiod
# import time

# #chip = gpiod.Chip("gpiochip0")
# chip = gpiod.chip("/dev/gpiochip0")
# step_pins = [22, 23, 24, 25]

# lines = chip.get_lines(step_pins)
# lines.request(consumer="stepper_test", type=gpiod.LINE_REQ_DIR_OUT)

# # try:
# #     while True:
# #         print("GPIO ON (HIGH)")
# #         lines.set_values([1, 1, 1, 1])  # 모든 핀 HIGH
# #         time.sleep(2)
# #         print("GPIO OFF (LOW)")
# #         lines.set_values([0, 0, 0, 0])  # 모든 핀 LOW
# #         time.sleep(2)
# # except KeyboardInterrupt:
# #     print("\n[INFO] Test stopped.")
# #     lines.release()


# # 하프 스텝 순서 (ULN2003 모듈에 맞게 설정)
# half_step_sequence = [
#     [1, 0, 0, 1],  # Step 1
#     [1, 0, 0, 0],  # Step 2
#     [1, 1, 0, 0],  # Step 3
#     [0, 1, 0, 0],  # Step 4
#     [0, 1, 1, 0],  # Step 5
#     [0, 0, 1, 0],  # Step 6
#     [0, 0, 1, 1],  # Step 7
#     [0, 0, 0, 1],  # Step 8
# ]

# def step_motor(steps, speed_factor=1.0):
#     """
#     steps: 회전할 스텝 수 (ex: 512 = 1바퀴)
#     speed_factor: 속도 조절 (1.0 = 기본 속도, 0.5 = 2배 빠름, 2.0 = 2배 느림)
#     """
#     base_delay = 0.002  # 기본 딜레이 (속도 기본값)
#     delay = base_delay * speed_factor  # 속도 조절

#     for _ in range(steps):
#         for step in half_step_sequence:
#             lines.set_values(step)
#             time.sleep(delay)
# try:
#     print("모터를 시계방향으로 회전 중...")
#     step_motor(512,speed_factor=0.4)  # 한 바퀴 회전 (28BYJ-48 기준)
    
#     time.sleep(1)
    
#     print("모터를 반시계방향으로 회전 중...")
#     step_motor(512,speed_factor=0.4)

# except KeyboardInterrupt:
#     print("\n프로그램 종료.")

# finally:
#     # GPIO 해제
#     lines.set_values([0, 0, 0, 0])
#     lines.release()

import lgpio
import time

# 0번 칩 열기 (보통 Raspberry Pi의 기본 GPIO 칩)
handle = lgpio.gpiochip_open(0)

# IN1~4에 해당하는 GPIO 핀 번호
pins = [14, 15, 18, 23]

# 각 핀을 출력 모드로 설정
for pin in pins:
    lgpio.gpio_claim_output(handle, pin)

# full-step 방식의 시퀀스 예제
# (모터에 따라 시퀀스가 달라질 수 있으니 필요시 조정하세요)
step_sequence = [
    (1, 0, 0, 0),
    (1, 1, 0, 0),
    (0, 1, 0, 0),
    (0, 1, 1, 0),
    (0, 0, 1, 0),
    (0, 0, 1, 1),
    (0, 0, 0, 1),
    (1, 0, 0, 1)
]

# 각 스텝 간의 딜레이 (초 단위)
delay = 0.0008
# 실행할 전체 스텝 수 (예: 100번의 스텝)
total_steps = 1000

# 스텝 시퀀스 반복
for i in range(total_steps):
    for step in step_sequence:
        lgpio.gpio_write(handle, pins[0], step[0])
        lgpio.gpio_write(handle, pins[1], step[1])
        lgpio.gpio_write(handle, pins[2], step[2])
        lgpio.gpio_write(handle, pins[3], step[3])
        time.sleep(delay)

# 사용 후 모든 핀을 LOW 상태로 전환하여 모터 정지
for pin in pins:
    lgpio.gpio_write(handle, pin, 0)

# 칩 닫기
lgpio.gpiochip_close(handle)


























































































import lgpio
import time

# GPIO 핀 번호 설정 (BCM 모드)
PWM_GPIO = 12 
CHIP_NUM = 0  # gpiochip0 사용

# GPIO 칩 핸들 열기
h = lgpio.gpiochip_open(CHIP_NUM)

# GPIO 핀을 PWM 출력으로 설정
lgpio.gpio_claim_output(h, PWM_GPIO)

# # # # 서보모터 원리 → PWM 개념 → 듀티 사이클 → lgpio.tx_pwm() 값 변환 방식
# # # # 일반 DC모터는 계속 회전하는데, 서보모터는 지정된 각도만큼 회전하고 멈추는 기능이 있음 이를 위해서는 PWM기능이 필요함
# # # # 서보모터는 PWM신호를 생성하는 핀에 연결하고, 펄스신호의 듀티 사이클(%)에 따라 특정 각도로 회전함
# # # # 펄스를 주는 동안 서보모터는 위치를 유지, 펄스를 멈추면 일부 서보모터는 상태를 유지하고, 일부는 초기 위치로 돌아감
# # # # 펄스가 HIGH로 유지되는 시간(마이크로초)를 기반으로
# # # # EX) 50HZ PWM 신호(1주기 = 20ms)에서, PWM주기가 (50ms), 그 중 HIGH(1)로 유지되는 시간만큼 움직임.
# # # # 펄스가 5%(500마이크로초) -> 0도
# # # # 펄스가 7.5%(1500마이크로초) -> 90도
# # # # 펄스가 10%(2500마이크로초) -> 180도                         -----------이건 각도제어 180 서보모터임. 내거는 아님-------
# # # #               ------------------서보모터는 펄스가 HIGH로 유지되는 시간(마이크로초)를 가지고 움직인다----------------
# # # # 50HZ PWM(1주기 = 20마이크로초) 안에서 HIGH로 유지되는 시간이 펄스지속시간             50HZ등의 값은 데이터시트로 확인, 사양 값인듯    
# # # # DUTY CYCLE(%) = (펄스지속시간) / PWM주기     * 100
# # # # DUTY CYCLE(%) = 5 + (각도 * 5) / 180 

def set_servo_angle(angle):
    """각도(0~180)를 PWM 듀티 사이클 (0~100%)로 변환"""
    if angle < 0 or angle > 180:
        raise ValueError("Angle must be between 0 and 180")
    
    # 서보모터의 PWM 듀티 사이클 범위 변환 (5%~10%)
    duty_cycle = (angle / 180.0) * 5 + 5  # 0도=5%, 180도=10%           %로 맞추기
    print("각도로 변환된 duty cycle: ", duty_cycle)
    print(f"Setting angle: {angle} -> PWM Duty Cycle: {duty_cycle}%")
    lgpio.tx_pwm(h, PWM_GPIO, 50, duty_cycle)  # 50Hz PWM 적용   
    # h:핸들
    # PWM_GPIO: 내가 연결한 pwm지원하는 핀
    # 50: 서보모터의 주기
    # duty_cycle: 원하는 각도를 기준으로 만들어진 듀티 사이클
#set_servo_angle(0)
#lgpio.tx_pwm(h, PWM_GPIO, 30, 7)


# set_servo_angle(170)  # 90도
# time.sleep(1)
set_servo_angle(0)
time.sleep(1)
set_servo_angle(170)  # 90도
time.sleep(1)
# try:
#     while True:
#         set_servo_angle(0)  # 0도
#         time.sleep(1)
#         set_servo_angle(90)  # 90도
#         time.sleep(1)
#         set_servo_angle(180)  # 180도
#         time.sleep(1)
# except KeyboardInterrupt:
#     lgpio.gpiochip_close(h)
#     print("서보모터 제어 종료")














































































import lgpio
import time

PWM_GPIO = 12  # GPIO 12번 핀 사용
CHIP_NUM = 0   # gpiochip0 사용

# GPIO 핸들 열기
h = lgpio.gpiochip_open(CHIP_NUM)

# PWM 출력을 위해 GPIO 핀을 출력 모드로 설정
lgpio.gpio_claim_output(h, PWM_GPIO)

def set_servo_speed(speed, hold_time=0):
    """
    360도 회전형 MG90S 서보모터 속도 제어
    speed: -100 (최대 반대 방향) ~ 0 (정지) ~ +100 (최대 정방향)
    """
    if speed < -100 or speed > 100:
        raise ValueError("속도는 -100 ~ +100 사이여야 합니다.")

    # 🚀 7.5% (정지) 기준으로 속도를 변환
    duty_cycle = 7.5 + (speed * 2.5 / 100)  # -100일 때 5%, 100일 때 10%

    print(f"속도 {speed} → PWM Duty Cycle: {duty_cycle}%")
    lgpio.tx_pwm(h, PWM_GPIO, 50, duty_cycle)  # 50Hz PWM 적용

    if hold_time > 0:
        time.sleep(hold_time)
        lgpio.tx_pwm(h, PWM_GPIO, 7.5, 50)  # 다시 정지
        print("정지")

try:
    print("정지")
    set_servo_speed(0, 2)   # 정지 (2초)

    print("정방향 최대 속도")
    set_servo_speed(50, 0.4)  # 최대 속도로 시계 방향 (3초)
    
    print("정지")
    set_servo_speed(0, 2)   # 정지 (2초)

    print("반대 방향 최대 속도")
    set_servo_speed(-50, 0.4)  # 최대 속도로 반시계 방향 (3초)
    
    print("정지")
    set_servo_speed(0, 2)   # 정지 (2초)

except KeyboardInterrupt:
    set_servo_speed(0)  # PWM 정지
    lgpio.gpiochip_close(h)
    print("서보모터 제어 종료")


# 360회전 서보모터는 각도가 아니라 속도랑 방향을 제어해야함
# 중립위치(정지) = 듀티사이클이 1500일때, 정지

# 1500마이크로초 미만일 때, 시계방향으로 회전 값이 작을수록 빠르게
# 초과일때, 반시계 회전, 값이 클 수록 빠르게