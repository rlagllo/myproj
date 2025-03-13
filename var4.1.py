from imutils import face_utils
import dlib
import cv2
import os
os.chdir("C:\\Users\\김해창\\Desktop\\VAR")

image = cv2.imread("person.jpg", cv2.IMREAD_COLOR) #opencv에서 이미지를 읽음, (주소, 컬러모드)
width, height, channel = image.shape
#회전하는게 아니고 회전행렬을 구하는 거임. 회전 시키려면 warpAffine으로 적용해야함
#rotated = cv2.getRotationMatrix2D((width/2, height/2), 270, 1) #사진 회전 (중심점 위치, 각도, 스케일링)
#image = cv2.warpAffine(image, rotated, (width, height)) #이미지변환 (원본, 행렬(회전), 사진크기)

sunglasses = cv2.imread("sunglasses.png", -1)  # 선글라스 -1은 투명도 채널로, 추가시켰음

p = "shape_predictor_68_face_landmarks.dat" #이미 훈련된 데이터
detector = dlib.get_frontal_face_detector() #dlib에서 제공, HOG라는 메서드 사용함 그런 얼굴탐지기 생성
predictor = dlib.shape_predictor(p) #dlib에서 제공하는 거 말고, 다른 이미 훈련된 데이터를 기반으로하는 랜드마크 예측모델 로드

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 이미지를 그레이스케일로 변환, 랜드마크 예측에는 컬러가 필요없음
faces = detector(gray)  #detector는 얼굴만 찾아냄

for face in faces: #찾아낸 얼굴에서 랜드마크 찾아낼 거임 faces: 사진 내의 모든 얼굴(로 판단된)의 리스트
    landmarks = predictor(gray, face)  # 얼굴 랜드마크 예측, 여기서 68개의 랜드마크 전부 만들어짐
    left_eye = (landmarks.part(36).x-50, landmarks.part(36).y+50)
    right_eye = (landmarks.part(45).x+50, landmarks.part(45).y+50)
# 37번 ~ 42번: 왼쪽 눈 (왼쪽 눈썹과 눈동자 주변)
# 43번 ~ 48번: 오른쪽 눈 (오른쪽 눈썹과 눈동자 주변)
    # 눈 사이의 거리 계산
    eye_width = int((right_eye[0] - left_eye[0])*1)
    eye_height = int(eye_width * 0.3)  # 선글라스 높이 비율 조정
    
    sunglasses_resized = cv2.resize(sunglasses, (eye_width, eye_height))
    
    for n in range(36, 48):  # 이미 만들어진 랜드마크 68개에 좌표를 지정, 눈 36~48
        x, y = landmarks.part(n).x, landmarks.part(n).y  #68개의 랜드마크 좌표들
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)  # 얼굴 랜드마크 그리기, 원으로

    # 얼굴 영역 자르기
    x1, y1, x2, y2 = (face.left(), face.top(), face.right(), face.bottom())  # 얼굴 좌표
    face_image = image[y1:y2, x1:x2]  # 얼굴 부분만 자르기
    
    y_offset = int(min(left_eye[1], right_eye[1]) - eye_height // 2)
    x_offset = left_eye[0]
    
    for c in range(0, 3):  # RGB 채널 (Red, Green, Blue) 순으로 처리
        image[y_offset:y_offset+eye_height, x_offset:x_offset+eye_width, c] = \
            sunglasses_resized[:, :, c] * (sunglasses_resized[:, :, 3] / 255.0) + \
            image[y_offset:y_offset+eye_height, x_offset:x_offset+eye_width, c] * (1.0 - sunglasses_resized[:, :, 3] / 255.0)
    # 잘라낸 얼굴 이미지 표시
    cv2.imshow("Cropped Face", face_image)

# 키 입력 대기 후 종료
cv2.waitKey(0)
cv2.destroyAllWindows()
# 27번: 코끝
# 28번 ~ 36번: 코 옆 (양쪽 콧구멍, 코 다리)
# 눈
# 37번 ~ 42번: 왼쪽 눈 (왼쪽 눈썹과 눈동자 주변)
# 43번 ~ 48번: 오른쪽 눈 (오른쪽 눈썹과 눈동자 주변)
# 입
# 49번 ~ 59번: 입 주위 (입술 모양)
# 60번 ~ 68번: 입꼬리 (윗입술, 아랫입술, 입술 모양)
# 눈썹
# 17번 ~ 22번: 왼쪽 눈썹 (눈썹 위쪽)
# 23번 ~ 27번: 오른쪽 눈썹 (눈썹 위쪽)
# 얼굴 윤곽
# 1번 ~ 17번: 얼굴 외곽 (턱선 포함)