import tensorflow as tf
m_new = tf.keras.models.load_model('abc.h5')
mnist =  tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
import cv2
import numpy as np
image = x_test[1]
input = cv2.resize(image, (28 , 28)).reshape((1,28 , 28))
print(m_new.predict_classes(input))
print(y_test[1])

import cv2
import numpy as np
img = np.ones((600,600), dtype='uint8') *255
# designating a 400 x 400 pixels point of interest on which digits will be drawn
img[100:500,100:500] = 0

windowName = 'Digits Project'
cv2.namedWindow(windowName)

abc=False

# mouse callback function
def draw_circle(event, x, y, flags, param):
    global abc

    if event == cv2.EVENT_LBUTTONDOWN:
        abc = True
        cv2.circle(img, (x, y), 10, (255, 255, 255), -1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if abc == True:
            cv2.circle(img, (x, y), 10, (255, 255, 255), -1)

    else:
        abc = False

# bind the callback function to window
cv2.setMouseCallback(windowName, draw_circle)
while (True):
    cv2.imshow(windowName, img)
    key = cv2.waitKey(2)
    if key == ord('q'):
        break
    elif key == ord('c'):
        img[100:500,100:500] = 0
    elif key == ord('p'):
        image = img[100:500,100:500]
        input = cv2.resize(image, (28 , 28)).reshape((1,28 , 28))
        print(m_new.predict_classes(input))
        

cv2.destroyAllWindows()
        

