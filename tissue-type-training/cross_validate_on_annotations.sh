ARGS="--magnification 20
--cohort_csv_path data/dataframes/tissuetype_hne_df.csv
--preprocessed_cohort_csv_path data/dataframes/preprocessed_tissuetype_hne_df.csv
--tile_dir pretilings
--tile_selection_type manual
--otsu_threshold 0.2
--batch_size 96
--min_n_tiles 1
--num_epochs 30 
--crossval 4
--gpu 0
--overlap 32
--tile_size 64
--normalize
--model resnet18
--experiment_name xval4
--learning_rate 0.0005"

python preprocess.py ${ARGS}

python pretile.py ${ARGS}

python train_tissue_tile_clf.py ${ARGS}

python pred_tissue_tile.py ${ARGS} --checkpoint_path "checkpoints/xval4_fold0_epoch021.torch" --val_pred_file "predictions/xval4_fold0_epoch021.torch_val.csv" 
python pred_tissue_tile.py ${ARGS} --checkpoint_path "checkpoints/xval4_fold1_epoch021.torch" --val_pred_file "predictions/xval4_fold1_epoch021.torch_val.csv" 
python pred_tissue_tile.py ${ARGS} --checkpoint_path "checkpoints/xval4_fold2_epoch021.torch" --val_pred_file "predictions/xval4_fold2_epoch021.torch_val.csv" 
python pred_tissue_tile.py ${ARGS} --checkpoint_path "checkpoints/xval4_fold3_epoch021.torch" --val_pred_file "predictions/xval4_fold3_epoch021.torch_val.csv" 

python eval_tissue_tile.py ${ARGS} --val_pred_file "predictions/xval4_fold{}_epoch021.torch_val.csv" 
