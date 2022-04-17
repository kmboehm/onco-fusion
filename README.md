# onco-fusion

```
code/
├── ct-feature-extraction  -- convert MHA/MHD CT files to radiomic features
│   ├── make_windowed_vols.py -- apply abdominal window to volumetric CT images
│   ├── params_left_ovary25.yaml
│   ├── params_omentum25.yaml
│   ├── params_right_ovary25.yaml
│   └── pyrad.py -- extract adnexal/omental features
├── feature_selection -- select features by prognostic relevance
│   ├── discover_omentum_feats.sh
│   ├── discover_ovary_feats.sh
│   ├── discover_path_feats.sh
│   └── select_features.py -- perform univariate Cox modeling for each feature; plot the resulting logHR and significance
├── hne-feature-extraction -- convert SVS whole-slide image files to intepretable histopathologic features
│   ├── extract_feats_from_bitmaps.py -- from resulting tissue type bitmaps, extract morphologic features
│   ├── extract_feats_from_object_detections.py -- from QuPath cell detections, extract morphologic and intensity-based features
│   ├── infer_tissue_tile_clf.py -- classify foreground tiles as belonging to fat, necrosis, tumor, or stroma
│   ├── map_inference_to_bitmap.py -- from tile classification, generate one unified bitmap
│   ├── qupath -- singularity container to run qupath-stardist pipeline headlessly
│   │   ├── data
│   │   ├── detections
│   │   │  └─ CMU-1-Small-Region_2_stardist_detections_and_measurements.tsv
│   │   ├── docker-compose.yml
│   │   ├── dockerfile
│   │   ├── init_singularity_env.sh
│   │   ├─ Makefile
│   │   ├── models
│   │   │   ├── ANN_StardistSeg3.0CellExp1.0CellConstraint_AllFeatures_LymphClassifier.json -- classify cells as lymphocyte or other
│   │   │   └── he_heavy_augment -- StarDist model to segment nuclei
│   │   │       ├── saved_model.pb
│   │   │       └── variables
│   │   │           ├── variables.data-00000-of-00001
│   │   │           └── variables.index
│   │   ├── README.md
│   │   └── scripts
│   │       └── stardist_nuclei_and_lymphocytes.groovy -- detect nuclei and classify as other or lymphocytes; extract morpholopgic and intensity-based features
│   ├─ run_cpu_qupath_stardist_singleSlide.sh -- run singularity pipeline on gpu
│   ├── run_gpu_qupath_stardist_singleSlide.sh
│   ├── tissue_tile_inference2.sh -- run pipeline to infer tissue types and extract features
  |     └── tissue_type_classifier_weights.torch -- trained weights for tissue type classifier
├── survival-modeling
│   ├── general.py -- utilities to support train_test.py
│   ├── mutual_feature_testing.py -- script used to test mutual information among modalities
│   ├── select_features.py -- script used to select clinical features
│   └── train_test.py -- main script to train and test models
└── tissue-type-training
    ├── config.py -- options for training
    ├── dataset.py -- dataset objects for loading tiles and classes
    ├── eval_tissue_tile.py -- evaluate peformance of tissue tile classifier
    ├── general_utils.py
    ├── models.py -- ResNet-18-derived models
    ├── preprocess.py -- determine foreground regions; note included tiles 
    ├── pretile.py -- generate directories of PNG images from whole-slide images for fast loading during training and inference
    ├── tissue_tile_clf_norm.sh -- run training pipeline
    └── train_tissue_tile_clf.py -- load data, train model, save parameters (either in cross-validation or full-dataset formulation)
```
