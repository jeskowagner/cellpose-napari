import pathlib

def extract_zip(cached_file, url, data_path):
    import zipfile
    from cellpose import utils

    if not cached_file.exists():
        utils.download_url_to_file(url, cached_file)        
        with zipfile.ZipFile(cached_file,"r") as zip_ref:
            zip_ref.extractall(data_path)

def _get_data_dir():
    cp_dir = pathlib.Path.home().joinpath('.cellpose')
    cp_dir.mkdir(exist_ok=True)
    zip_path = cp_dir.joinpath('data.zip')
    data_dir = cp_dir.joinpath('data')
    extract_zip(zip_path, 'https://osf.io/download/s52q3/', cp_dir)
    return data_dir


def make_rgb_2d_sample():
    from cellpose.io import imread

    data_dir = _get_data_dir()
    data = imread(str(data_dir / '2D' / 'rgb_2D_tif.tif'))
    return [(data, {'name': 'Cells (2D+2Ch)'})]


def make_rgb_3d_sample():
    from cellpose.io import imread

    data_dir = _get_data_dir()
    data = imread(str(data_dir / '3D' / 'rgb_3D.tif')) # shape (Z, C, Y, X)
    # to get display in napari with channels, use channel_axis=1

    return [(data, {'name': 'Cells (3D+2Ch)', "axis_labels": ("z", "c", "y", "x")})]
