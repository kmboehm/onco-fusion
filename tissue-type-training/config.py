import argparse
from torch import device
import yaml
import os


with open('../global_config.yaml', 'r') as f:
    CONFIGS = yaml.safe_load(f)
    DATA_DIR = CONFIGS['data_dir']

parser = argparse.ArgumentParser()

parser.add_argument('--preprocessed_cohort_csv_path',
                    type=str,
                    default='preprocessed_msk_os_h&e_cohort.csv',
                    help='Full path to CSV file describing whole slide images and outcomes.')

parser.add_argument('--checkpoint_path',
                    type=str,
                    default='',
                    help='Location of checkpoint to load for preds OR to resume for training.')

parser.add_argument('--experiment_name',
                    type=str,
                    default='EXPERIMENT_NAME',
                    help='name under which to store checkpoints')

parser.add_argument('--crossval',
                    type=int,
                    default=0,
                    help='0: no xval, >0: k-fold xval')

parser.add_argument('--normalize',
                    action='store_true',
                    default=False,
                    help='whether to apply macenko normalization')

parser.add_argument('--cohort_csv_path',
                    type=str,
                    default='msk_os_h&e_cohort.csv',
                    help='Full path to CSV file describing whole slide images and outcomes.')

parser.add_argument('--tile_size',
                    type=int,
                    default=512,
                    help='Edge length of each tile used for training/evaluation.')

parser.add_argument('--tile_selection_type',
                    type=str,
                    default='manual',
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

parser.add_argument('--batch_size',
                    type=int,
                    default=96,
                    help='Batch size for inference or training.')

parser.add_argument('--gpu',
                    type=int,
                    default=[0],
                    nargs='+',
                    help='Which GPU(s) to use for training.')

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
                    default='pretilings_512',
                    help='Directory to store/load tiles..')

parser.add_argument('--val_pred_file',
                    type=str,
                    default='',
                    help='Location of val pred .csv to load for eval.')


args = parser.parse_args()

args.preprocessed_cohort_csv_path = os.path.join(DATA_DIR, args.preprocessed_cohort_csv_path)
args.cohort_csv_path = os.path.join(DATA_DIR, args.cohort_csv_path)

desired_otsu_thumbnail_tile_size = 4
