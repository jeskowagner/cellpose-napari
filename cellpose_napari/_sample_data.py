import os
import pathlib


def _load_cellpose_data(image_name, dname):
    import numpy as np
    from cellpose.io import imread
    from cellpose.utils import download_url_to_file

    cp_dir = pathlib.Path.home().joinpath('.cellpose')
    cp_dir.mkdir(exist_ok=True)
    data_dir = cp_dir.joinpath('data')
    data_dir.mkdir(exist_ok=True)
    data_dir_3D = data_dir.joinpath('3D')
    data_dir_3D.mkdir(exist_ok=True)

    url = 'https://www.cellpose.org/static/data/' + image_name
    cached_file = str(data_dir_3D.joinpath(image_name))
    if not os.path.exists(cached_file):
        download_url_to_file(url, cached_file, progress=True)
    data = imread(cached_file)
    if '3D' in image_name:
        data = np.moveaxis(data, 0, 1)
    return [(data, {'name': dname})]


def make_rgb_3d_sample():
    return _load_cellpose_data('rgb_3D.tif', 'Cells (3D+2Ch)')