# Set up

## Download Synapse repository
https://www.synapse.org/#!Synapse:syn25946117/wiki/611576

## Download H&E WSIs
Download H&E WSIs listed within the downloaded Synapse repository at `data/hne/tcga/manifest.txt` using GDC Data Transfer Tool (https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Data_Download_and_Upload/). Ensure a flat file structure (.svs files within the `data/hne/tcga` folder).

## Clone this GitHub repository
It is recommended to clone the GitHub repository into the same directory as the Synapse repository. Conda environments are provided as `environment.yml` packages for each module of the pipeline.

## Set global parameters
In `global_config.yaml`, set the full paths to the directories enclosing the data and code. All scripts assume that the code and data are within subdirectories of these paths, enitled `code` and `data` respectively.

## Move Singularity image to code repository
Move `qupath-stardist_latest.sif` from `data` to `code/hne-feature-extraction/qupath`.

# Tissue type training
Using annotations by gynecologic pathologists (found within the `tissue-type-training` directory of the Synapse repository), train a semantic segmentation model to infer tissue type from H&E images. This component is optional: the resulting weights of our training are already stored in `tissue_type_training/checkpoints/tissue_type_classifier_weights.torch`. Other than paths set in global YAML file in the previous step, all options are set in `config.py`. For help, use `python config.py --h`.

## Cross-validate model for tissue type inference
 `tissue_type_training/cross_validate_on_annotations.sh`
Use this to explore various model types and hyperparameter configurations.

## Train model for tissue type inference
`tissue_type_training/train_on_all_annotations.sh` Note that `preprocess.py` and `pretile.py` must be run before this step (a sufficient example is in the cross validation script, so running that before this is sufficient).
 
# H&E feature extraction
## Extract tissue type features
Next, we apply our trained model to semantically segment tissue types on slides from our multimodal patient cohort: `hne-feature-extraction/1_infer_tissue_types_and_extract_features.sh`. This is a deterministic process and should exactly replicate the tissue type-based features in `reference_hne_features.csv`.

## Identify nuclei
Using the StarDist extension for QuPath, we perform instance segmentation of cellular nuclei and apply a bespoke classification script to distinguish lymphocytes from other nuclei: `hne-feature-extraction/2_extract_objects.sh`. Before running this script, move or copy slides of interest from `data/hne` to `code/hne-feature-extraction/qupath/data/slides`.

## Label nuclei by tissue type; extract nuclear features
Finally, we coregister the two feature spaces and extract descriptive statistics for nuclei of each cell type: `hne-feature-extraction/3_label_objects_and_extract_features.sh`. We observe that nuclear detection varies somewhat by run with StarDist, and thus the object-based features may vary among runs. We observe on the order of 1% variation in features calculated in the aggregate (e.g., mean tumor nuclear size).


# CT feature extraction
We apply the abdominal window and extract features from omental and adnexal lesions contoured by fellowship-trained diagnostic radiologists: `ct-feature-extraction/extract_ct_features.sh`. Features are stored as a csv file in the features subdirectory.


# Feature selection 
We use log partial hazard ratios and their associated significance calculated on univariate Cox regression to select informative features from the CT and H&E feature spaces: `feature-selection/select_features.sh`. The log partial hazard ratio for each feature across the training cohort and the associated volcano plot is generated for each modality in the results subdirectory.


# Survival modeling
Use `feature-selection/environment.yml`.
We build univariate survival models for histopathologic, radiologic, clinical, and genomic information spaces. Subsequently, we combine the modalities in a late fuson framework and plot the performance: `survival-modeling/train_test.py`. Relevant results and figures are generated in the respective subdirectories.
