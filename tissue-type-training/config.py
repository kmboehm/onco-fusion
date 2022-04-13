import argparse
from torch import device

parser = argparse.ArgumentParser()

# make_thumbnails

parser.add_argument('--thumbnail_dir',
                    type=str,
                    default='/gpfs/mskmind_ess/boehmk/histocox/annotation_thumbnails',
                    help='Full path to directory for thumbnails.')

# general

parser.add_argument('--preprocessed_cohort_csv_path',
                    type=str,
                    default='/gpfs/mskmind_ess/boehmk/histocox/preprocessed_msk_os_h&e_cohort.csv',
                    help='Full path to CSV file describing whole slide images and outcomes.')

parser.add_argument('--checkpoint_path',
                    type=str,
                    default='',
                    help='Location of checkpoint to load for preds OR to resume for training.')


parser.add_argument('--crossval',
                    type=int,
                    default=0,
                    help='0: no xval, -1: MCCV, >0: k-fold xval, -2: no validation set')

parser.add_argument('--subsample',
                    type=int,
                    default=0,
                    help='whether to subsample tiles: 0 does not subsample')

parser.add_argument('--normalize',
                    action='store_true',
                    default=False,
                    help='whether to apply macenko normalization')

# for preprocess.py

parser.add_argument('--cohort_csv_path',
                    type=str,
                    default='/gpfs/mskmind_ess/boehmk/histocox/msk_os_h&e_cohort.csv',
                    help='Full path to CSV file describing whole slide images and outcomes.')

parser.add_argument('--wsi_dir',
                    type=str,
                    default='/gpfs/mskmind_ess/pathology_images/ov',
                    help='Full path to directory of whole slide images.')

parser.add_argument('--tile_size',
                    type=int,
                    default=512,
                    help='Edge length of each tile used for training/evaluation.')

parser.add_argument('--tile_selection_type',
                    type=str,
                    default='manual',
                    help='manual or otsu or lab')

parser.add_argument('--annotation_dir',
                    type=str,
                    default='/gpfs/mskmind_ess/boehmk/histocox/annotation_results',
                    help='manual or otsu')

parser.add_argument('--otsu_threshold',
                    type=float,
                    default=0.25,
                    help='Percentage foreground required to include tile.')

parser.add_argument('--purple_threshold',
                    type=float,
                    default=0.25,
                    help='Percentage purple required to include tile.')

parser.add_argument('--magnification',
                    type=int,
                    default=20,
                    help='Magnification of WSI.')

parser.add_argument('--model',
                    type=str,
                    default='resnet18',
                    help='cnn architecture')

parser.add_argument('--overlap',
                    type=int,
                    default=0,
                    help='n pixels of tile overlap (for preprocess and pretile)')

# for outcome_mil_train.py
parser.add_argument('--mil_training_batch_size',
                    type=int,
                    default=96,
                    help='Batch size for MIL training.')

parser.add_argument('--mil_training_extreme',
                    type=str,
                    default='min',
                    help="'min' or 'max', designating slide-wise tile selection for training")

parser.add_argument('--mil_training_n',
                    type=int,
                    default=1,
                    help="number of slide-wise tiles to select for training")

parser.add_argument('--mil_training_skip_first_pass_on_all',
                    action='store_true',
                    default=False,
                    help='whether to train on entire dataset in first epoch')

# for biomarker_train.py and outcome_train.py

parser.add_argument('--batch_size',
                    type=int,
                    default=96,
                    help='Batch size for inference and all-tile training.')

parser.add_argument('--slide_limit',
                    type=int,
                    default=-1,
                    help='Max number of slides per patient to be used for training/validation. -1 is no limit.')

parser.add_argument('--gpu',
                    type=int,
                    default=[0],
                    nargs='+',
                    help='Which GPUs to use for training.')

parser.add_argument('--slide_code',
                    type=str,
                    default=[],
                    nargs='+',
                    help='Which slide codes to include for training ("TS" and "BS" are frozen; "DX" is FFPE).')

parser.add_argument('--learning_rate',
                    type=float,
                    default=0.001,
                    help='Learning rate for training.')

parser.add_argument('--weight_decay',
                    type=float,
                    default=1e-4,
                    help='Weight decay for Adam optimizer.')

parser.add_argument('--num_epochs',
                    type=int,
                    default=20,
                    help='Number of epochs to train.')

parser.add_argument('--min_n_tiles',
                    type=int,
                    default=100,
                    help='Minimum number of tiles per slide.')

parser.add_argument('--tile_dir',
                    type=str,
                    default='/gpfs/mskmind_ess/boehmk/histocox/pretilings_512',
                    help='Directory to store/load tiles..')

# for eval.py

parser.add_argument('--val_pred_file',
                    type=str,
                    default='',
                    help='Location of val pred .csv to load for eval.')

parser.add_argument('--test_pred_file',
                    type=str,
                    default='',
                    help='Location of test pred .csv to load for eval.')

parser.add_argument('--train_pred_file',
                    type=str,
                    default='',
                    help='Location of train pred .csv to load for eval.')

parser.add_argument('--visualize',
                    action='store_true',
                    default=False)

parser.add_argument('--base_path',
                    type=str,
                    default='/gpfs/mskmind_ess/boehmk/histocox')



args = parser.parse_args()

if args.checkpoint_path:
    assert 'epoch' in args.checkpoint_path

desired_otsu_thumbnail_tile_size = 4