import cv2 as cv
from models.experimental import attempt_load
from detect import detect
import requests

def post_image(url, token, image, without_amount, total):
    data = {
        'camera_token': token,
        'without_amount': without_amount,
        'image': image,
        'total': total
    }
    r = requests.post(url, data=data)


if __name__ == '__main__':
    weights, img_size, device, save = './models/best.pt', 384, 'cpu', './results'
    model = attempt_load(weights, map_location=device)

    TOKEN = "68c7bb4780b495b5ee20641b00d586ae"
    URL = "https://tsmc.gnsjhenjie.tech/api/cameraIncident"
    FREQ = 100

    cnt = 0
    camera = cv.VideoCapture(0)
    while camera.isOpened():
        cnt += 1
        status, frame = camera.read()
        cv.imshow("Frame", frame)
        if cnt == FREQ:
            without_amount, total = detect(frame, save, model, device, img_size)
            if without_amount > 0:
                #post_image(URL, TOKEN, frame, without_amount, total)
                pass
            cnt = 0

        key = cv.waitKey(1)
        # ESC
        if key == 27:
            break
    camera.release()
    cv.destroyAllWindows()
