import subprocess
import pyautogui
from subprocess import PIPE
import time
#import docker

times = 50
#f = open("./cursor_position_for_2-3.txt","w",encoding='UTF-8')

datalist = []
for i in range(times):
    print("{} 枚目".format(i))
    time.sleep(2)
    subprocess.run(['gnome-screenshot', '-p'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    x, y = pyautogui.position()
    data = "x座標: " + str(x) + " | " + "y座標: " + str(y) + "\n"
    datalist.append(data)

for i in range(len(datalist)):
    f.write(datalist[i])
