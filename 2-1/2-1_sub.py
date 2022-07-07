import rospy
from std_msgs.msg import String
import csv

position_list = []
f = open('./2.csv', 'w')
writer = csv.writer(f)

def process_image(msg):
        writer.writerow(msg.data.split("|"))
        
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