ARGS="--magnification 20
--tile_dir pretilings
--cohort_csv_path data/dataframes/tissuetype_hne_df.csv
--preprocessed_cohort_csv_path data/dataframes/preprocessed_tissuetype_hne_df.csv
--tile_selection_type manual
--otsu_threshold 0.2
--batch_size 96
--min_n_tiles 1
--num_epochs 22
--crossval 0
--gpu 0
--overlap 32
--tile_size 64
--normalize
--model resnet18
--experiment_name fulldata
--learning_rate 0.0005"

python train_tissue_tile_clf.py ${ARGS}
