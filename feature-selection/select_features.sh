python select_features.py code/ct-feature-extraction/features/ct_features_omentum.csv --outcome_df_path data/dataframes/clin_df.csv --output_df_path results/hr_ct_features_omentum.csv --output_plot_path results/hr_ct_features_omentum.png --modality radiology --method cph --train_id_df_path data/dataframes/train_ids.csv

python select_features.py code/ct-feature-extraction/features/ct_features_ovary.csv --outcome_df_path data/dataframes/clin_df.csv --output_df_path results/hr_ct_features_ovary.csv --output_plot_path results/hr_ct_features_ovary.png --modality radiology --method cph --train_id_df_path data/dataframes/train_ids.csv

python select_features.py code/hne-feature-extraction/tissue_tile_features/reference_hne_features.csv --outcome_df_path data/dataframes/clin_df.csv --output_df_path results/hr_hne_features.csv --output_plot_path results/hr_hne_features.png --modality pathology --method cph --xy_covar n_foreground_tiles --train_id_df_path data/dataframes/train_ids.csv

