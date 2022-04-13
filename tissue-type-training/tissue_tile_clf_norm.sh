BASE_PATH='/gpfs/mskmind_ess/boehmk/histocox'
ARGS="--magnification 20
--annotation_dir ${BASE_PATH}/tissue_type_annotations
--cohort_csv_path ${BASE_PATH}/20200818_msk_spectrum_slide_viewer_annotations.csv
--preprocessed_cohort_csv_path ${BASE_PATH}/preprocessed_20200818_msk_spectrum_slide_viewer_annotations_128_overlap_edited20210204.csv
--tile_dir ${BASE_PATH}/pretilings_128_20x_75percentOverlap_normalized
--tile_selection_type manual
--otsu_threshold 0.2
--batch_size 96
--min_n_tiles 1
--num_epochs 30 
--crossval 4
--gpu 1
--overlap 32
--tile_size 64
--normalize
--model resnet18
--learning_rate 0.0005"

#python get_annotations.py ${ARGS}
#python preprocess.py ${ARGS}
python pretile.py ${ARGS}
python train_tissue_tile_clf.py ${ARGS}
#python visualize_tile_label_pairs.py ${ARGS}
