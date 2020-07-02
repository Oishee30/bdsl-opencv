from fastai.vision import load_learner,open_image
import cv2
import numpy as np

learn = load_learner('E:\\STUDY\\fastai\\Pytorch\\Detection\\models','export.pkl')
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
   
    a = cv2.resize(frame, (224,224), interpolation = cv2.INTER_AREA)
    
    cv2.imwrite('a.png',a)
    
    a = open_image('a.png')
    
    pred_class = 0
    font = cv2.FONT_HERSHEY_SIMPLEX 
    pred_class,pred_idx,outputs = learn.predict(a)
    if(outputs[pred_idx] < 0.4):
        pred_class = 'None'
    cv2.putText(frame,  
                str(pred_class),  
                (50, 50),  
                font, 1,  
                (0, 255, 255),  
                2,  
                cv2.LINE_4) 
    cv2.imshow("frame", frame)
  #  print(pred_class)
    
    key = cv2.waitKey(1)
    if key == 27 :
        break
cap.release()
cv2.destroyAllWindows()