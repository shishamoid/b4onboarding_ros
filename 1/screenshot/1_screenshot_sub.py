import rospy
from sensor_msgs.msg import Image 
import cv2
import ros_numpy

def process_image(msg):
    try:
        np_img = ros_numpy.numpify(msg)
        cv2.imwrite('./screenshot.jpg',np_img)

    except Exception as err:
        print(err)

def start_node():
    rospy.init_node('operator')
    rospy.loginfo('img_proc node started')
    rospy.Subscriber("image_data", Image, process_image)
    rospy.spin()

if __name__ == '__main__':
    try:
        start_node()
    except rospy.ROSInterruptException:
        pass