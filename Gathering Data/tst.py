#Abhiraj Abhaykumar Eksambekar
#abhiraj@pdx.edu
#generate data set for hand gestures

import cv2
import os
import time

lp = True

FolderName = input("Please Enter Your Name: ")
path = '{}/'.format(FolderName)
dir1= path + 'zero/'
dir2= path + 'one/'
dir3= path + 'two/'
dir4= path + 'three/'
dir5= path + 'four/'
dir6= path + 'five/'
dir7= path + 'spider_man/'
dir8= path + 'rock/'
dir9= path + 'thumbsup/'
dir10= path + 'ok/'
if not os.path.exists(dir1):
    os.makedirs(dir1)
if not os.path.exists(dir2):
    os.makedirs(dir2)
if not os.path.exists(dir3):
    os.makedirs(dir3)
if not os.path.exists(dir4):
    os.makedirs(dir4)
if not os.path.exists(dir5):
    os.makedirs(dir5)
if not os.path.exists(dir6):
    os.makedirs(dir6)
if not os.path.exists(dir7):
    os.makedirs(dir7)
if not os.path.exists(dir8):
    os.makedirs(dir8)
if not os.path.exists(dir9):
    os.makedirs(dir9)
if not os.path.exists(dir10):
    os.makedirs(dir10)

cam = cv2.VideoCapture(1)
ret, frame = cam.read()
if not ret:
    cam.release()
    cam = cv2.VideoCapture(1)

image_counter = 0
gesture_counter = 0
colorss = (255,0,0)
thickness = 2

print("Thank you for helping me creating a data-set for my project. ")
print("We will record 10 hand signs and 100 images of each gesture")
print("Starting from number signs- zero, one, to five, and some extra signs-spiderman, rock, ok, thumbsup")
print("Reference image will be shown to follow")
print("Place your hand in the blue rectangle")
discard = input("Press enter when ready");

def overlap(emoji):
    img2 = cv2.imread(emoji)
    rows,cols,channels = img2.shape
    # roi = frame[0:rows,int(shape_fr[1]-cols):int(shape_fr[1])]
    # mask_me = cv2.bitwise_or(roi,img2)
    # frame[0:rows,int(shape_fr[1]-cols):int(shape_fr[1])] = cv2.bitwise_and(roi,img2)
    frame[0:rows,int(shape_fr[1]-cols):int(shape_fr[1])] = img2

def handCheck():
    cv2.putText(frame, "Place your hand in blue and hit Spacebar", (20,25),cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,0,0))

def gesFun(nxt):
    global image_counter
    if image_counter != 200:
        image_name = "{}.png".format(image_counter)
        cv2.imwrite(path + image_name, new_frame)
        image_counter += 1
        time.sleep(0.050)
    else:
        cv2.putText(frame, "Hit Spacebar To Continue, Next is {}".format(nxt), (20,25),cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,0,0))

def br():
    global lp
    lp = False

while lp:
    ret, frame = cam.read()
    if not ret:
        print("Camera not Detected!")
        break
    shape_fr = frame.shape
    start_pt = (int(shape_fr[1]/16),int(shape_fr[0]/4))
    end_pt = (int(shape_fr[1]/2-shape_fr[1]/16),int(shape_fr[0] - shape_fr[0]/4))

    new_frame = frame[int(shape_fr[0]/4):int(shape_fr[0] - shape_fr[0]/4),int(shape_fr[1]/16):int(shape_fr[1]/2-shape_fr[1]/16)]
    frame = cv2.rectangle(frame, start_pt, end_pt, colorss, thickness)

    if gesture_counter == 0:
        handCheck()
    elif gesture_counter == 1:
        path = dir1
        overlap('emoji/fist.jpg')
        gesFun("one")
    elif gesture_counter == 2:
        path = dir2
        overlap('emoji/one.jpg')
        gesFun("two")
    elif gesture_counter == 3:
        path = dir3
        overlap('emoji/two.jpg')
        gesFun("three")
    elif gesture_counter == 4:
        path = dir4
        overlap('emoji/three.jpg')
        gesFun("four")
    elif gesture_counter == 5:
        path = dir5
        overlap('emoji/four.jpg')
        gesFun("five")
    elif gesture_counter == 6:
        path = dir6
        overlap('emoji/five.jpg')
        gesFun("spiderman")
    elif gesture_counter == 7:
        path = dir7
        overlap('emoji/spiderman.jpg')
        gesFun("rock sign")
    elif gesture_counter == 8:
        path = dir8
        overlap('emoji/rock.jpg')
        gesFun("thumb up")
    elif gesture_counter == 9:
        path = dir9
        overlap('emoji/tup.jpg')
        gesFun("ok")
    elif gesture_counter == 10:
        path = dir10
        overlap('emoji/ok.jpg')
        gesFun("done Thank you!")
    elif gesture_counter == 11:
        br()

    cv2.putText(frame, "{}".format(image_counter), (int(shape_fr[0]/2 + 50),int(shape_fr[1]/2)),cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,255,0))

    cv2.imshow("test",frame)
    cv2.imshow("cropp frame",new_frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        print("Escape pressed")
        break
    elif k %256 == 32:
        gesture_counter += 1
        image_counter = 0

cam.release()
cv2.destroyAllWindows()

print("Thanks again, I really appreciate your help!! :)")
time.sleep(5)
