import cv2
import time
import requests

cap = cv2.VideoCapture("https://open.ys7.com/v3/openlive/BD3202818_1_2.m3u8?expire=1740641527&id=814882500458319872&t=455106fe74399439794cf3fe4ba7e25ab9b0c55feb5b8216502d26236d208154&ev=100")
cap.set(cv2.CAP_PROP_FPS, 30)


device_sn = [
    "BD3202818",
    "BD0632093",
    "BE6992578",
    "BC8197657",
    "BE3337110",
]




if __name__ == "__main__":
    while cap.isOpened():
        ret, frame = cap.read()
        current_time = time.time()
        local_time = time.localtime(current_time)
        formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', local_time)
        if not ret or frame is None or frame.size == 0:
            raise ValueError("frame is invalid!")
        else:
            print(f"------------------------------------{formatted_time}")