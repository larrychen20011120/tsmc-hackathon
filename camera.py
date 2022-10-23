import cv2 as cv
from models.experimental import attempt_load
from detect import detect
import requests
import time
import sys
from playsound import playsound

def post_image(url, token, private, img_name, without_amount, total):
    data = {
        'camera_token': token,
        'without_amount': without_amount,
        'total': total
    }
    if not private:
        files = [
            ('image', (img_name+'.jpg', open(f'./results/{img_name}.jpg', 'rb'), 'image/jpeg'))
        ]
    else:
        files = [
            ('image', None)
        ]
    r = requests.request("POST", url+'/cameraIncident', headers={}, data=data, files=files)
    print('Post Image:', r.text)

def post_active(url, token, status='active'):
    data = {
        'camera_token': token,
        'status': status
    }
    r = requests.request("POST", url+'/camera', headers={}, data=data)
    print('POST Active', r.text)

if __name__ == '__main__':
    # conf, iou, private
    if len(sys.argv) > 2:
        conf = float(sys.argv[1])
        iou = float(sys.argv[2])
        private = bool(sys.argv[3])
    else:
        iou = 0.45
        conf = 0.25
        private = False
    weights, img_size, device, save = './models/best.pt', 384, 'cpu', './results'
    model = attempt_load(weights, map_location=device)

    TOKEN = "68c7bb4780b495b5ee20641b00d586ae"
    URL = "https://tsmc.gnsjhenjie.tech/api"
    MAX_SIZE = 30
    ACTIVE = 600
    POST = 300
    DETECT = 50

    start_time = time.time()
    send_time = 0
    cnt = 0

    camera = cv.VideoCapture(0)
    while camera.isOpened():
        cnt += 1
        status, frame = camera.read()
        duration = time.time() - start_time
        if duration >= ACTIVE:
            post_active(URL, TOKEN, status='active')
            start_time = time.time()
        if cnt == DETECT:
            cnt = 0
            without_amount, total = detect(frame, save, model, device, img_size, conf, iou)
            if without_amount > 0:
                current = time.time()
                playsound('./results/chinese.wav')
                playsound('./results/taiwanese.wav')
                playsound('./results/english.mp3')
                if current - send_time >= POST:
                    post_image(URL, TOKEN, private, 'result', without_amount, total)
                    send_time = current

    camera.release()
