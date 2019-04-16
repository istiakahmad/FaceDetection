import cv2
import time


class Camera:
    def __init__(self):
        w = 640  # Frame width...
        h = 480  # Frame hight...
        fps = 20.0  # Frames per second...
        resolution = (w, h)  # Frame size/resolution...

        self.video_capture = cv2.VideoCapture(0)  # Prepare the camera...
        print("Camera warming up ...")
        time.sleep(1)
        # Prepare Capture
        self.ret, self.frame = self.video_capture.read()

def main():
    # Create a camera instance...
    cam1 = Camera()

    while True:
        # Display the resulting frames...
        cam1.captureVideo()  # Live stream of video on screen...
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return ()

if __name__ == '__main__':
    main()
