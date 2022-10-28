
"""
# Description:
  Resize x2, x4, x8 and then crop to 256x256 patches.
"""
import os
import numpy as np
from PIL import Image
from pathlib import Path
import time


def random_crop(img_path, patch_size):
    """
    Randomly crop image to size of patch_size.
    :param img_path: string, path to image to be cropped.
    :param patch_size: list or array, (width, height)
    :return:
    """
    try:
        img = Image.open(img_path)
    except OSError:
        print("Error!")
        return None
    w, h = img.size

    imgs = [img]
    for r in [2, 4, 8]:
        wr = w // r
        hr = int(round(wr / w * h))
        # im = img.resize(size=(wr, hr), resample=Image.LANCZOS)
        im = img.resize(size=(wr, hr), resample=Image.NEAREST)
        imgs.append(im)
    # num_patches = [6, 6, 6, 6]
    num_patches = [12, 12, 12, 12]

    patch_width, patch_height = patch_size
    crop_list = []
    for k, N in enumerate(num_patches):
        img = imgs[k]
        wr, hr = img.size
        # print(f"k = {k}, Image size {wr:>4d}x{hr:>4d}, randomly crop {N} patches.")
        if wr > patch_width and hr > patch_height:
            for i in range(N):
                x0 = np.random.randint(wr - patch_width + 1)
                y0 = np.random.randint(hr - patch_height + 1)
                try:
                    img_crop = img.crop((x0, y0, x0 + patch_width, y0 + patch_height))
                except OSError:
                    return None
                crop_list.append(img_crop)
    if crop_list:
        return crop_list
    else:
        return None


def get_crop_name(path, postfix):
    name = os.path.basename(path).split('.')[0] + '_' + postfix + '.png'
    return name


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_imgs(path):
    img_list = list(Path(path).glob("*.png"))
    return img_list


if __name__ == '__main__':

    out_path = "./cropped_training_set"
    mkdir(out_path)
    img_list = get_imgs("./JPEG-AI-Dataset/training_set")
    print(f"Total number of images {len(img_list)}.")

    # Random crop
    counter = 0
    width, height = 256, 256
    start = time.time()
    for i, file in enumerate(img_list):

        cropped_images = random_crop(file, (width, height))

        if cropped_images:
            for j, f in enumerate(cropped_images):
                out_file = os.path.join(out_path, get_crop_name(file, f'x{2**(j//12)}_{j%12:02d}'))
                f.save(out_file)

            counter += len(cropped_images)
            # print(f"\tActual number of cropped patches is {len(cropped_images)}.")

        if i % 100 == 99:
            elapsed_time = time.time() - start
            hour = int(elapsed_time // 3600)
            mins = int((elapsed_time - hour * 3600) // 60)
            secs = int(round(elapsed_time % 60))
            print(f'Processed {i + 1} images. | Elapsed time {hour} hr, {mins} min, {secs} sec.')

    print(f"Cropped to {counter} patches from {len(img_list)} images.")
