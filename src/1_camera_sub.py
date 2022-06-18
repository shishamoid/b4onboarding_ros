import rospy
from sensor_msgs import msg
from sensor_msgs.msg import Image 
import ros_numpy
import cv2


def process_image(msg):
    try:
        np_img = ros_numpy.numpify(msg)
        cv2.imwrite('./camera.jpg', np_img)

    except Exception as err:
        print(err)

def start_node():
    rospy.init_node('image_view')
    rospy.Subscriber("image_raw", Image, process_image)
    rospy.spin()

if __name__ == '__main__':
    try:
        start_node()
    except rospy.ROSInterruptException:
        pass