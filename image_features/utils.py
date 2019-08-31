import multiprocessing
import pretrainedmodels.utils as utils
from tqdm import tqdm

load_img = utils.LoadImage()


def _is_valid_img(img_path):
    try:
        load_img(img_path)
        return True
    except Exception:
        return False


def filter_invalid_images(img_paths, num_workers=4, progress=False):
    """Filter invalid images before computing expensive features."""
    with multiprocessing.Pool(num_workers) as p:
        if progress:
            load_works = list(tqdm(
                p.imap(_is_valid_img, img_paths),
                total=len(img_paths),
                desc="Filtering invalid images"))
        else:
            load_works = p.map(_is_valid_img, img_paths)

    img_paths = [
        img_path for img_path, is_loadable in
        zip(img_paths, load_works) if is_loadable
    ]
    return img_paths
