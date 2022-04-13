BASE_PATH='/gpfs/mskmind_ess/boehmk/histocox'
ARGS="--magnification 20
--annotation_dir ${BASE_PATH}/tissue_type_annotations
--cohort_csv_path ${BASE_PATH}/20210325_multimodal_nact_slides.csv
--preprocessed_cohort_csv_path ${BASE_PATH}/preprocessed_20210325_multimodal_nact_slides_128.csv
--tile_dir ${BASE_PATH}/pretilings_128_20x
--wsi_dir /gpfs/mskmind_ess/pathology_images/ov_by_image_id
--tile_selection_type otsu
--otsu_threshold 0.5
--purple_threshold 0.05
--batch_size 200
--min_n_tiles 1
--gpu 0 1 2
--tile_size 128
--model resnet18
--checkpoint_path ${BASE_PATH}/checkpoints/2021-02-06_14.26.33_fold-2_epoch018.torch"

#python preprocess.py ${ARGS}
#python pretile.py ${ARGS}
#python infer_tissue_tile_clf.py ${ARGS}
#python map_inference_to_bitmap.py ${ARGS}
python extract_feats_from_bitmaps.py ${ARGS}
