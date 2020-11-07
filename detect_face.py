import time, cv2
from  datetime import datetime
from csv import DictWriter
import os
import os.path
face_cs= cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_frontalface_default.xml')

video = cv2.VideoCapture(0)

a1=time.time()
a=1
while True:
    check,frame= video.read()
    #gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    picc= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces= face_cs.detectMultiScale(picc,scaleFactor = 1.2, minNeighbors=5)
    for x,y,w,h in faces:
        img1= cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),3)
    cv2.imshow("Video",img1)
    a+=1
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
    
a2=time.time()
a3=a2-a1
fps=a/a3
if os.path.exists('My Experiments')==False:
    os.mkdir('My Experiments')
    
with open("My Experiments\Data.csv",'a',newline='') as cam:
    dict_writ=DictWriter(cam, fieldnames=["No. of frames","Time","fps"])
    if os.stat('My Experiments\Data.csv').st_size==0:
        dict_writ.writeheader()
    dict_writ.writerow({"No. of frames":a,"Time":a3,"fps":fps})
    
    
print(f"Number of Frames:{a}\ntime taken:{a3}second\nNo.of frames per second:{fps} fps ")
video.release()
cv2.destroyAllWindows()
time.sleep(5)
