課題1
１git clone
2 cd
1 docker イメージをダウンロード
2 コンテナをビルド

docker run --device=/dev/video0:/dev/video0 --network host -e DISPLAY=$DISPLAY -v /home/itodaisuke/zemi/b4onboarding_ros:/root -v /tmp/X11-unix:/tmp/X11-unix:rw -it ros:melodic3




![ros](/home/itodaisuke/ピクチャ/ros.png)