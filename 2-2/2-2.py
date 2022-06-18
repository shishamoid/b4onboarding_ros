import subprocess
import pyautogui
from subprocess import PIPE
import time

times = 50

datalist = []
for i in range(times):
    print("{} 枚目".format(i))
    time.sleep(2)
    subprocess.run(['gnome-screenshot', '-p'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #↑screenshotはデフォルトで/home/<username>/ピクチャに吐かれるみたいです。
    x, y = pyautogui.position()
    data = "x座標: " + str(x) + " | " + "y座標: " + str(y) + "\n"
    datalist.append(data)

with open("../2-3/cursor_position_for_2-3.txt","w") as f:
    for i in range(len(datalist)):
        f.write(datalist[i])
