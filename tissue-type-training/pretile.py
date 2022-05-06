import general_utils
import config
import os
import numpy as np
import yaml

from PIL import Image
from openslide import OpenSlide
from openslide.lowlevel import OpenSlideUnsupportedFormatError
from openslide.deepzoom import DeepZoomGenerator
from joblib import delayed, Parallel
from skimage.color import rgb2lab, lab2rgb

from general_utils import make_otsu


# normalization tools from https://github.com/CODAIT/deep-histopath/blob/master/deephistopath/preprocessing.py
stain_ref = np.array([[0.56237296, 0.38036293],
       [0.72830425, 0.83254214],
       [0.39154767, 0.40273766]])
max_sat_ref = np.array([[0.62245465],
       [0.44427557]])

beta = 0.15
alpha = 1
light_intensity = 255


# credit: StainTools (https://github.com/Peter554/StainTools/blob/master/staintools/preprocessing/luminosity_standardizer.py)
def get_standard_luminosity_limit(rgb):
    assert isinstance(rgb, np.ndarray)
    lab = rgb2lab(rgb)
    p = np.percentile(lab[:, :, 0], 95)
    return p


# credit: StainTools (https://github.com/Peter554/StainTools/blob/master/staintools/preprocessing/luminosity_standardizer.py)
def apply_standard_luminosity_limit(rgb, p):
    assert isinstance(rgb, np.ndarray)
    lab = rgb2lab(rgb)
    lab[:, :, 0] = np.clip(100 * lab[:, :, 0] / p, 0, 100)
    return np.round(np.clip(255 * lab2rgb(lab), 0, 255)).astype(np.uint8)

# credit: StainTools (https://github.com/Peter554/StainTools/blob/master/staintools/preprocessing/luminosity_standardizer.py)
def calculate_macenko_transform(to_transform):
    assert isinstance(to_transform, np.ndarray)

    c = to_transform.shape[2]
    assert c == 3

    luminosity_limit = get_standard_luminosity_limit(to_transform)
    to_transform = apply_standard_luminosity_limit(to_transform, luminosity_limit)

    im = rgb2lab(to_transform)
    ignore_mask = ((im[:, :,0] < 50) & (np.abs(im[:,:,1] - im[:,:,2]) < 30)) | (im[:,:,2] < -40) | (im[:,:,0] > 90) | (im[:,:,1] > 40)
    to_transform = to_transform[~ignore_mask, :]

    to_transform = to_transform.reshape(-1, c).astype(np.float64)  # shape (H*W, C)

    # Convert RGB to OD.
    OD = -np.log10(to_transform/light_intensity + 1e-8)

    # Remove data with OD intensity less than beta.
    OD_thresh = OD[np.all(OD >= beta, 1), :]

    # Calculate eigenvectors.
    U, s, V = np.linalg.svd(OD_thresh, full_matrices=False)

    # Extract two largest eigenvectors.
    top_eigvecs = V[0:2, :].T * -1  # shape (C, 2)

    # Project thresholded optical density values onto plane spanned by
    # 2 largest eigenvectors.
    proj = np.dot(OD_thresh, top_eigvecs)  # shape (K, 2)

    # Calculate angle of each point wrt the first plane direction.
    # Note: the parameters are `np.arctan2(y, x)`
    angles = np.arctan2(proj[:, 1], proj[:, 0])  # shape (K,)

    # Find robust extremes (a and 100-a percentiles) of the angle.
    min_angle = np.percentile(angles, alpha)
    max_angle = np.percentile(angles, 100 - alpha)

    # Convert min/max vectors (extremes) back to optimal stains in OD space.
    # This computes a set of axes for each angle onto which we can project
    # the top eigenvectors.  This assumes that the projected values have
    # been normalized to unit length.
    extreme_angles = np.array(
        [[np.cos(min_angle), np.cos(max_angle)],
         [np.sin(min_angle), np.sin(max_angle)]]
    )  # shape (2,2)
    stains = np.dot(top_eigvecs, extreme_angles)  # shape (C, 2)

    # Merge vectors with hematoxylin first, and eosin second, as a heuristic.
    if stains[0, 0] < stains[0, 1]:
        stains[:, [0, 1]] = stains[:, [1, 0]]  # swap columns

    # Calculate saturations of each stain.
    # Note: Here, we solve
    #    OD = VS
    #     S = V^{-1}OD
    # where `OD` is the matrix of optical density values of our image,
    # `V` is the matrix of stain vectors, and `S` is the matrix of stain
    # saturations.  Since this is an overdetermined system, we use the
    # least squares solver, rather than a direct solve.
    sats, _, _, _ = np.linalg.lstsq(stains, OD.T, rcond=None)

    # Normalize stain saturations to have same pseudo-maximum based on
    # a reference max saturation.
    max_sat = np.percentile(sats, 99, axis=1, keepdims=True)
    return stains, max_sat, luminosity_limit


# credit: StainTools (https://github.com/Peter554/StainTools/blob/master/staintools/preprocessing/luminosity_standardizer.py)
def apply_macenko_transform(stains, max_sat, luminosity_limit, to_transform):
    assert isinstance(to_transform, np.ndarray)

    h, w, c = to_transform.shape
    assert c == 3

    to_transform = apply_standard_luminosity_limit(to_transform, luminosity_limit)

    to_transform = to_transform.reshape(-1, c).astype(np.float64)  # shape (H*W, C)

    # Convert RGB to OD.
    OD = -np.log10(to_transform/light_intensity + 1e-8)

    # Calculate saturations of each stain.
    # Note: Here, we solve
    #    OD = VS
    #     S = V^{-1}OD
    # where `OD` is the matrix of optical density values of our image,
    # `V` is the matrix of stain vectors, and `S` is the matrix of stain
    # saturations.  Since this is an overdetermined system, we use the
    # least squares solver, rather than a direct solve.
    sats, _, _, _ = np.linalg.lstsq(stains, OD.T, rcond=None)

    # Normalize stain saturations to have same pseudo-maximum based on
    # a reference max saturation.
    sats = sats / max_sat * max_sat_ref

    # Compute optimal OD values.
    OD_norm = np.dot(stain_ref, sats)

    # Recreate image.
    # Note: If the image is immediately converted to uint8 with `.astype(np.uint8)`, it will
    # not return the correct values due to the initial values being outside of [0,255].
    # To fix this, we round to the nearest integer, and then clip to [0,255], which is the
    # same behavior as Matlab.
    # x_norm = np.exp(OD_norm) * light_intensity  # natural log approach
    x_norm = 10 ** (-OD_norm) * light_intensity - 1e-8  # log10 approach
    x_norm = np.clip(np.round(x_norm), 0, 255).astype(np.uint8)
    x_norm = x_norm.astype(np.uint8)
    x_norm = x_norm.T.reshape(h, w, c)
    return x_norm


def pretile_slide(row, tile_size, tile_dir, normalize=False, overlap=0):
    slide_dir = os.path.join(tile_dir, row['image_path'].split('/')[-1].replace('.svs', ''))
    if os.path.exists(slide_dir):
        n_tiles_saved = len(os.listdir(slide_dir))
        if n_tiles_saved == row.n_foreground_tiles:
            print("{} fully tiled; skipping".format(slide_dir))
            return
    else:
        os.mkdir(slide_dir)
    
    slide = OpenSlide(row['image_path'])

    slide_mag = general_utils.get_magnification(slide)
    if slide_mag == config.args.magnification:
        level_offset = 0
    elif slide_mag == 2 * config.args.magnification:
        level_offset = 1
    elif slide_mag == 4 * config.args.magnification:
        level_offset = 2
    else:
        raise NotImplementedError

    if normalize:
        size0, size1 = slide.dimensions
        stains, max_sat, luminosity_limit = calculate_macenko_transform(
            np.array(slide.get_thumbnail((size0//16, size1//16))))
    else:
        stains, max_sat, luminosity_limit = None, None, None

    addresses = row.tile_address
    generator, level = general_utils.get_full_resolution_generator(slide, tile_size=tile_size,
                                                                   level_offset=level_offset,
                                                                   overlap=overlap)
    for address, class_ in addresses:
        tile_file_name = os.path.join(slide_dir,
                    str(address).replace('(', '').replace(')', '').replace(', ', '_') + '.png')
        if os.path.exists(tile_file_name):
            continue
        tile = generator.get_tile(level, address)

        if normalize:
            tile = apply_macenko_transform(stains, max_sat, luminosity_limit, np.array(tile))
            tile = Image.fromarray(tile)
        tile.save(tile_file_name)


if __name__ == '__main__':
    serial = True
    df = general_utils.load_preprocessed_df(file_name=config.args.preprocessed_cohort_csv_path,
                                            min_n_tiles=1,
                                            explode=False)

    with open('../global_config.yaml', 'r') as f:
        DIRECTORIES = yaml.safe_load(f)
        DATA_DIR = DIRECTORIES['data_dir']
    df['image_path'] = df['image_path'].apply(lambda x: os.path.join(DATA_DIR, x))
    
    if serial:
        for _, row in df.iterrows():
            pretile_slide(row=row,
                          tile_size=config.args.tile_size,
                          tile_dir=config.args.tile_dir,
                          normalize=config.args.normalize,
                          overlap=config.args.overlap)
    else:
        Parallel(n_jobs=64)(delayed(pretile_slide)(row=row,
                          tile_size=config.args.tile_size,
                          tile_dir=config.args.tile_dir,
                          normalize=config.args.normalize,
                          overlap=config.args.overlap)
                           for _, row in df.iterrows())
