import rospy
from std_msgs.msg import String

position_list = []

def process_image(msg):
    f.write(msg.data + "\n")
        
def start_node():
    rospy.init_node('operator')
    rospy.loginfo('img_proc node started')
    rospy.Subscriber('position_data', String, process_image)
    rospy.spin()

if __name__ == '__main__':
    try:
        start_node()
    except rospy.ROSInterruptException:
        pass