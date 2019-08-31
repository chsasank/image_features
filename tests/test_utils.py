import numpy as np
import os
from image_features.utils import filter_invalid_images


data_path = os.path.join(os.path.dirname(__file__), 'data')


def test_image_features():
    example_imgs = [
        os.path.join(data_path, 'example_image.jpg'),
        os.path.join(data_path, 'example_image_2.JPG'),
        os.path.join(data_path, 'example_image_corrupted.JPG')
    ]

    valid_imgs = filter_invalid_images(example_imgs)
    assert valid_imgs == example_imgs[:2]

if __name__ == '__main__':
    test_image_features()
