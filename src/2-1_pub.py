import rospy

from std_msgs.msg import String
import pyautogui

position_list = []

def operator():
    rospy.init_node('operator', anonymous=True)
    pub = rospy.Publisher('position_data', String, queue_size=10)
    rate = rospy.Rate(1) 
    while not rospy.is_shutdown():
        x, y = pyautogui.position()
        data = "x座標: " + str(x) + " | " + "y座標: " + str(y)
        pub.publish(data)
        rate.sleep()

if __name__ == '__main__':
    try:
        #pass
        operator()
    except rospy.ROSInterruptException:
        pass
