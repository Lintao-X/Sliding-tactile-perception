import cv2
import numpy as np
import os
import time
from simple_lama_inpainting import SimpleLama
from PIL import Image




def auto_stitching(scene_path, template_path, roi, num_pic, locations_x, locations_y):
    scene = cv2.imread(scene_path, 0)
    scene_test = scene.copy()
    template = cv2.imread(template_path, 0)
    i = num_pic

    print("num_pic is:", i)
    if scene is None or template is None:
        print("Error: Could not open or find the image.")
        return
    h_template, w_template = template.shape
    h_scene, w_scene = scene.shape
    scene_half = scene[:, :w_scene//2 + 100]

    blur_scene = cv2.GaussianBlur(scene, (5, 5), 0)
    scene_half1 = cv2.adaptiveThreshold(blur_scene, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 33, 3)

    y0, x0, h, w = roi
    y1 = y0 + h
    x1 = x0 + w

    template_crop = template[y0:y1, x0:x1]

    blur = cv2.GaussianBlur(template_crop, (5, 5), 0)
    template_crop1 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 33, 3)

    result = cv2.matchTemplate(scene_half1, template_crop1, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    print("max_val is:", max_val)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(scene_test, top_left, bottom_right, (0, 255, 0), 2)

    abs_x = abs(top_left[0] - x0)
    abs_y = abs(top_left[1] - y0)

    if abs_x // (1 + i) < w_template // (3 + i) or max_val < 0.1:
        os.remove(scene_path)
        print("removed before stitching")
        return (None,None,None)
    else:


        del_roi_y = top_left[1] - y0
        del_img_y = h_template - h_scene
        if top_left[1] - y0 < 0:
            if del_img_y>abs_y:
                stitched_image = np.full((h_template, w_scene + abs_x), 255, dtype=np.uint8)
            else:
                stitched_image = np.full((h_scene + abs_y, w_scene + abs_x), 255, dtype=np.uint8)

            stitched_image[:h_template, :w_template] = template
            stitched_image[abs_y:abs_y + h_scene, abs_x:abs_x + w_scene] = scene
        else:
            print("scene is below")
            stitched_image = np.full((h_template + abs_y, w_scene + abs_x), 255, dtype=np.uint8)
            h_stitched, w_stitched = stitched_image.shape
            stitched_image[abs_y:h_stitched, :w_template] = template
            stitched_image[:h_scene, abs_x:abs_x + w_scene] = scene

        return stitched_image, abs_x, abs_y


def find_roi(binary_image, width_scene):
    height, width = binary_image.shape
    roi_height = 340 # modify according to your needs
    roi_width = 180
    scale_factor = 0.2
    width_scene *= scale_factor

    downsampled_height = int(height * scale_factor)
    downsampled_width = int(width * scale_factor)
    downsampled_image = cv2.resize(binary_image, (downsampled_width, downsampled_height), interpolation=cv2.INTER_LINEAR)

    max_black_pixels = 0
    best_roi_downsampled = None
    for i in range(0, downsampled_height - int(roi_height * scale_factor) + 1):
        for j in range(int(downsampled_width - width_scene / 2), downsampled_width - int(roi_width * scale_factor) + 1):
            downsampled_roi = downsampled_image[i:i + int(roi_height * scale_factor), j:j + int(roi_width * scale_factor)]
            black_pixels = np.sum(downsampled_roi == 0)
            if black_pixels > max_black_pixels:
                max_black_pixels = black_pixels
                best_roi_downsampled = (i, j, int(roi_height * scale_factor), int(roi_width * scale_factor))

    if best_roi_downsampled is not None:
        start_y, start_x, downsampled_height, downsampled_width = best_roi_downsampled
        start_y_index = int(start_y / scale_factor)
        start_x_index = int(start_x / scale_factor)
        roi_height_index = roi_height
        roi_width_index = roi_width
        if start_x_index < width // 2:
            start_x_index = width // 2
        max_black_pixels = 0
        best_roi = None
        for i in range(max(0, start_y_index - 10), min(height - roi_height_index, start_y_index + 10)):
            for j in range(max(width // 2, start_x_index - 10), min(width, start_x_index + roi_width_index + 10)):
                roi = binary_image[i:i + roi_height_index, j:j + roi_width_index]
                black_pixels = np.sum(roi == 0)
                if black_pixels > max_black_pixels:
                    max_black_pixels = black_pixels
                    best_roi = (i, j, roi_height_index, roi_width_index)

        if best_roi is not None:
            start_y, start_x, roi_height, roi_width = best_roi
            cv2.rectangle(binary_image, (start_x, start_y), (start_x + roi_width, start_y + roi_height), 0, 2)
            return best_roi
    return None


def auto_threshold(image_path):
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    height, width = gray_image.shape

    mask = np.zeros((height, width), dtype=np.uint8)

    mask[gray_image == 255] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


    mask = cv2.dilate(mask, kernel, iterations=5)

    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    adaptive_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 33, 3)


    combined_mask = cv2.bitwise_or(mask, adaptive_thresh)

    return combined_mask



def process_subfolder(subfolder):
    image_files = sorted(os.listdir(subfolder))
    flag = 1
    num = 0
    num_pic = 0
    locations_x = []
    locations_y = []


    for i in range(len(image_files) - 1):
        print("num=", num)
        if flag:
            current_image_path = os.path.join(subfolder, image_files[i])
            next_image_path = os.path.join(subfolder, image_files[i + 1])
        else:
            current_image_path = os.path.join(subfolder, image_files[i - num])
            next_image_path = os.path.join(subfolder, image_files[i + 1])

        print('i=', i)
        print('next_image_path:', next_image_path)

        scene = cv2.imread(next_image_path, 0)
        height_scene, width_scene = scene.shape
        binary_image = auto_threshold(current_image_path)
        roi = find_roi(binary_image, width_scene)
        if roi is not None:
            template_path = current_image_path
            scene_path = next_image_path
            fused_image,abs_x, abs_y = auto_stitching(scene_path, template_path, roi, num_pic, locations_x, locations_y)
            if fused_image is not None:
                os.remove(next_image_path)
                cv2.imwrite(os.path.join(subfolder, image_files[i + 1]), fused_image)
                locations_x.append(abs_x)
                locations_y.append(abs_y)
                cv2.destroyAllWindows()
                flag = 1
                num = 0
                num_pic += 1
            else:
                cv2.destroyAllWindows()
                flag = 0
                num += 1
        else:
            print(f"No ROI found in {current_image_path}, skipping to next image.")


    image_files = sorted(os.listdir(subfolder))
    width_scene = 723
    height_scene = 723


    if image_files:
        last_image_path = os.path.join(subfolder, image_files[-1])
        if os.path.exists(last_image_path):
            last_image = cv2.imread(last_image_path, 0)
            if last_image is not None:
                height_last, width_last = last_image.shape

                mask = np.zeros((height_last, width_last), dtype=np.uint8)

                mask[last_image == 255] = 255

                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


                mask = cv2.dilate(mask, kernel, iterations=3)
                for abs_x, abs_y in zip(locations_x, locations_y):
                    x_center = abs_x
                    y_center = abs_y


                    mask[y_center - 15:y_center + 15, x_center - 15:x_center + width_scene] = 255
                    mask[y_center - 15:y_center + height_scene, x_center - 15:x_center + 15] = 255


                cv2.imwrite(os.path.join(subfolder, 'mask.png'), mask)
                print("Mask saved successfully.")

                simple_lama = SimpleLama()
                with Image.open(last_image_path) as img:

                    img_rgb = img.convert('RGB')

                result = simple_lama(img_rgb, mask)


                result.save(os.path.join(subfolder, 'inpaint.png'))

                file_path = os.path.join(subfolder, 'inpaint.png')

                image = cv2.imread(file_path, cv2.IMREAD_COLOR)


                hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


                h, s, v = cv2.split(hsv_image)


                output_path = os.path.join(subfolder, 'inpaint_v.png')
                cv2.imwrite(output_path, v)


            else:
                print(f"Failed to read the last image at {last_image_path}")
        else:
            print(f"Last image path {last_image_path} does not exist")
    else:
        print("No images found in the folder after processing.")

