import pandas as pd
import os
from PIL import Image 
import numpy as np
from joblib import Parallel, delayed
import sys
sys.path.append('../tissue-type-training')
import config


def visualize_cell_detections(df_, size, fn):
    marker_size = 2
    max_cells = 1000
    arr = np.zeros(size).astype(np.uint8)
    arr[:,:] = 0
    if len(df_) > max_cells:
        df = df_.sample(max_cells)
    else:
        df = df_.copy()
    for index, row in df.iterrows():
        arr[(row.CentroidY-marker_size):(row.CentroidY+marker_size), (row.CentroidX-marker_size):(row.CentroidX+marker_size)] = 255
    img = Image.fromarray(arr)
    img.save(fn)

    # now visualize by class
    map_ = {'Tumor': np.array([0, 255, 0]),
           'Stroma': np.array([0, 191, 255]),
           'Necrosis': np.array([255, 0, 0]),
           'Fat': np.array([255, 255, 0]),
           'Unknown': np.array([255, 255, 255])}
    arr = np.zeros((size[0], size[1], 3)).astype(np.uint8)
    for index, row in df.iterrows():
        if row.Parent == 'Unknown':
            continue
        arr[(row.CentroidY-marker_size):(row.CentroidY+marker_size), (row.CentroidX-marker_size):(row.CentroidX+marker_size)] = map_[row.Parent]
    img = Image.fromarray(arr)
    img.save(fn.replace('.png', '_classified.png'))


def load_tissue_maps(_dir, _classes):
    bbox_map = None
    d = {}
    for class_ in _classes:
        img_fn = os.path.join(_dir, class_ + '.png')
        if not os.path.exists(img_fn):
            return None, None
        img = Image.open(img_fn)
        img = np.array(img)
        img = (img != 0).astype(np.uint8)
        if bbox_map is None:
            bbox_map = img.astype(bool)
        else:
            bbox_map = bbox_map.astype(bool) | img.astype(bool)
        d[class_] = img
        img_size = img.shape
        #print(d[class_].shape)
    #print(np.argmin(bbox_map, axis=0).min())
    #print(np.argmax(bbox_map, axis=0).max())
    #print(np.argmin(bbox_map, axis=1).min())
    #print(np.argmax(bbox_map, axis=1).max())
    return d, img_size


def load_cell_detections(fn, scale_factor, shape):
    object_detections = pd.read_csv(fn, delimiter='\t')
    #print(object_detections)
    object_detections = object_detections.rename(columns={list(object_detections.columns)[5]: 'CentroidX', list(object_detections.columns)[6]: 'CentroidY'})
    object_detections['CentroidX'] =(object_detections['CentroidX'] // scale_factor).astype(int)
    object_detections['CentroidY'] = (object_detections['CentroidY'] // scale_factor).astype(int)
    #print(object_detections['CentroidX'].describe())
    #print(object_detections['CentroidY'].describe())
    object_detections.loc[object_detections.CentroidX >= shape[1], 'CentroidX'] = shape[1] - 1
    object_detections.loc[object_detections.CentroidY >= shape[0], 'CentroidY'] = shape[0] - 1
    object_detections['Parent'] = 'Unknown'
    return object_detections


def assign_cell_parents(object_detections, tissue_regional_maps):
    for class_, tissue_map in tissue_regional_maps.items():
        #print(class_)
        object_detections = object_detections.apply(lambda x: _assign_single_class(x, tissue_map, class_), axis=1)
    return object_detections


def _assign_single_class(row, tissue_map, tissue_type_name):
    tissue_map_val = tissue_map[row.CentroidY, row.CentroidX]
    if tissue_map_val != 0:
        row['Parent'] = tissue_type_name
    return row

def process_slide(slide_id):
    tissue_maps, img_size = load_tissue_maps(os.path.join(region_detection_dir, slide_id.split('.')[0]), classes)
    output_fn = os.path.join(output_dir, slide_id + '.csv') 
    if os.path.exists(output_fn):
        print("{} exists; skipping".format(output_fn))
        return

    if tissue_maps is None:
        print('{} has no tissue_maps; skipping'.format(slide_id))
        return
    
    if True: #try:
        cell_detections = load_cell_detections(os.path.join(object_detection_dir, slide_id + '.tsv'), object_coords_to_region_coords_scale_factor, img_size)
        cell_detections = assign_cell_parents(cell_detections, tissue_maps)
        #print(cell_detections.Parent.value_counts())
        print('processed {}'.format(slide_id))
        visualize_cell_detections(cell_detections, img_size, os.path.join(VIZ_DIR, slide_id + '.png'))
        cell_detections = cell_detections[cell_detections.Parent != 'Unknown']
        cell_detections.to_csv(output_fn, index=False)


checkpoint_id = config.args.checkpoint_path.split('/')[-1].replace('.torch', '')
object_detection_dir = 'qupath/data/results'
region_detection_dir = 'bitmaps/{}'.format(checkpoint_id)

slide_ids = [x[:-4] for x in os.listdir(object_detection_dir) if '.tsv' in x]
classes = ['Tumor', 'Stroma', 'Fat', 'Necrosis']

# bitmaps are generated at 4/128 = 1/32 resolution in pixel coordinates
# cells are detected in µmcoordinates at full resolution. 1 pixel = 05um
object_coords_to_region_coords_scale_factor = 16.096

output_dir = 'final_objects/{}'.format(checkpoint_id)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

VIZ_DIR = 'visualizations/{}'.format(checkpoint_id)
if not os.path.exists(VIZ_DIR):
    os.mkdir(VIZ_DIR)


if __name__ == '__main__':
    Parallel(n_jobs=64)(delayed(process_slide)(slide_id) for slide_id in slide_ids)
    #for slide_id in slide_ids:
    #    process_slide(slide_id)
