#!/bin/bash
source /gpfs/mskmind_ess/limr/mambaforge/etc/profile.d/conda.sh
conda activate transformer

ARGS="--magnification 20
--cohort_csv_path data/dataframes/hne_df.csv
--preprocessed_cohort_csv_path data/dataframes/preprocessed_hne_df.csv
--tile_dir ../tissue-type-training/pretilings
--tile_selection_type otsu
--otsu_threshold 0.5
--purple_threshold 0.05
--batch_size 200
--min_n_tiles 100
--normalize
--gpu 0 1 2 3
--tile_size 128
--model resnet18
--checkpoint_path ../tissue-type-training/checkpoints/tissue_type_classifier_weights.torch"

python ../tissue-type-training/preprocess.py ${ARGS}
python ../tissue-type-training/pretile.py ${ARGS}
python infer_tissue_tile_clf.py ${ARGS}
python map_inference_to_bitmap.py ${ARGS}
python extract_feats_from_bitmaps.py ${ARGS}
