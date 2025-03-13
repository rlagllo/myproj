from imutils import face_utils
import dlib
import cv2
import os
os.chdir("C:\\Users\\김해창\\Desktop\\VAR")
image = cv2.imread("person.jpg", cv2.IMREAD_COLOR) #opencv에서 이미지를 읽음, (주소, 컬러모드)

height, width, channel = image.shape
matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 1) #사진 회전 (중심점 위치, 각도, 스케일링)
dst = cv2.warpAffine(image, matrix, (width, height)) #이미지변환 (원본, 행렬(회전), 사진크기)

cv2.imshow("image", image)
cv2.imshow("dst", dst)
cv2.waitKey(0)
# You can close the window by pressing any key.
cv2.destroyAllWindows()
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0) #캠 사진 찍기
 
while True:
    # load the input image and convert it to grayscale
    _, image = cap.read()  #캠 찍은 거 저장
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #그레이스케일로 변환
        
    # detect faces in the grayscale image
    rects = detector(gray, 0)
    
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
    
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    
    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()