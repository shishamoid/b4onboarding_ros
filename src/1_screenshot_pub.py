import rospy
from sensor_msgs.msg import Image
import cv2

from PIL import ImageGrab
import numpy as np

import ros_numpy

def operator():
    rospy.init_node('operator', anonymous=True)
    pub = rospy.Publisher('image_data', Image, queue_size=10)

    img = ImageGrab.grab(bbox=(152, 206, 1752, 1406))
    imp = np.array(img)
    img = cv2.cvtColor(imp, cv2.COLOR_BGR2RGB)

    msg = ros_numpy.msgify(Image,img,encoding="bgr8")

    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        operator()
    except rospy.ROSInterruptException:
        pass
