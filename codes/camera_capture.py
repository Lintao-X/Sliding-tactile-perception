from multiprocessing import Process, Queue
from pypylon import pylon
import cv2
import time

def capture_frames(frame_queue):
    # connect to your camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())


    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    converter = pylon.ImageFormatConverter()

    # convert to OpenCV BGR
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    start_time = time.time()
    frame_count = 0

    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():

            image = converter.Convert(grabResult)
            img = image.GetArray()

            if not frame_queue.full():
                frame_queue.put(img)
                frame_count += 1

            # 每秒钟更新帧率
            if (time.time() - start_time) > 1:
                print(f"Current frames per second: {frame_count}")
                start_time = time.time()
                frame_count = 0

        grabResult.Release()


    camera.StopGrabbing()
    camera.Close()

if __name__ == '__main__':
    frame_queue = Queue(maxsize=10)
    capture_process = Process(target=capture_frames, args=(frame_queue,))
    capture_process.start()
    capture_process.join()
