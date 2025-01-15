from recording_contours import process_frames
from camera_capture import capture_frames
import os
import time
from multiprocessing import Process, Queue
import integrated_without_exhibition
import Predict



def clear_directory(directory):

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)
    print(f"Cleared directory: {directory}")

def main():
    save_dir = '/your_dir'
    frame_queue = Queue(maxsize=10)


    clear_directory(save_dir)


    capture_process = Process(target=capture_frames, args=(frame_queue,))
    process_process = Process(target=process_frames, args=(frame_queue, save_dir))

    capture_process.start()
    process_process.start()


    time.sleep(10)


    capture_process.terminate()
    process_process.terminate()


    capture_process.join()
    process_process.join()


    integrated_without_exhibition.process_subfolder(save_dir)

    Predict.predict(save_dir)


if __name__ == '__main__':
    main()