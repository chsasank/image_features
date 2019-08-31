import numpy as np
import os
from image_features import image_features


data_path = os.path.join(os.path.dirname(__file__), 'data')


def test_image_features():
    example_imgs = [
        os.path.join(data_path, 'example_image.jpg'),
        os.path.join(data_path, 'example_image_2.JPG')
    ]

    ftrs = image_features(example_imgs)
    expected_ftrs = np.load(os.path.join(data_path, 'ftrs.npy'))
    assert ftrs.shape == (2, 2048)
    assert np.allclose(ftrs, expected_ftrs)

if __name__ == '__main__':
    test_image_features()
