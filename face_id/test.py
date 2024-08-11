import cv2
from api import FaceRecognition
face_recognizer = FaceRecognition(path='ckpt/face_model/model_face.pt', device='cpu', mirror=True,mode='resnet_v2')
img = 'YOUR/PATH/For/TEST'
img = cv2.imread(img)
face_embd = face_recognizer.predict(img=img, meta=None)