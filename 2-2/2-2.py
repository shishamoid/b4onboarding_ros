import subprocess
import mouse
from subprocess import PIPE
import time
import csv

times = 100

datalist = []
for i in range(times):
    print("{} 枚目".format(i))
    time.sleep(2)
    subprocess.run(['gnome-screenshot', '-p'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #↑screenshotはデフォルトで/home/<username>/ピクチャに吐かれるみたいです。
    x, y = mouse.get_position()
    data =[x,y]
    datalist.append(data)

with open("../2-3/cursor_position_for_2-3.csv","w") as f:
    writer = csv.writer(f)
    for i in range(len(datalist)):
            writer.writerow(datalist[i])