#!/bin/bash

source /gpfs/mskmind_ess/limr/mambaforge/etc/profile.d/conda.sh
conda activate transformer
ARGS="
--preprocessed_cohort_csv_path data/dataframes/preprocessed_hne_df.csv
--checkpoint_path ../tissue-type-training/checkponts/tissue_type_classifier_weights.torch"

python merge_cells_and_regions.py ${ARGS}
python extract_feats_from_object_detections.py ${ARGS}
