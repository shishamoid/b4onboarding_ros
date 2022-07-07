from pdb import post_mortem
import rospy

from std_msgs.msg import String
import mouse
import csv

position_list = []

def operator():
    rospy.init_node('operator', anonymous=True)
    pub = rospy.Publisher('position_data', String, queue_size=10)
    rate = rospy.Rate(1) 
    while not rospy.is_shutdown():
        x, y = mouse.get_position() 
        data = str(x) +"|" +  str(y)
        pub.publish(data)
        rate.sleep()

if __name__ == '__main__':
    try:
        operator()        
    except rospy.ROSInterruptException:
        pass