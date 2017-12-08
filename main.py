import argparse
import os

from models.cyclegan import CycleGAN

def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        error = '{:s} is an invalid positive int value'.format(value)
        raise argparse.ArgumentTypeError(error)
    return ivalue

def positive_or_zero_int(value):
    ivalue = int(value)
    if ivalue < 0:
        error = '{:s} must be an int greater than or equal to 0'.format(value)
        raise argparse.ArgumentTypeError(error)
    return ivalue

def restricted_int(value):
    ivalue = int(value)
    if ivalue < 256:
        error = '{:s} must be an int greater than or equal to 256'.format(value)
        raise argparse.ArgumentTypeError(error)
    return ivalue

def positive_float(value):
    fvalue = float(value)
    if fvalue <= 0:
        error = '{:s} is an invalid positive float value'.format(value)
        raise argparse.ArgumentTypeError(error)
    return fvalue

def parse():
    parser = argparse.ArgumentParser(description='CycleGAN arguments.')
    
    #General arguments
    ge_args = parser.add_argument_group('general', 'General arguments')
    
    ge_args.add_argument('-u', '--min', action='store', nargs='?', 
                         default=-137.28571428571004, type=float, 
                         help=('Minimum voxel intensity. Default: '
                               '-137.28571428571004'))
    ge_args.add_argument('-v', '--max', action='store', nargs='?', 
                         default=17662.535068643472, type=float, 
                         help=('Maximum voxel intensity. Default: '
                               '17662.535068643472'))
    ge_args.add_argument('-n', '--name', action='store', nargs='?', 
                         default='gad', type=str, 
                         help='Dataset name. Default: "gad"')
    ge_args.add_argument('-t', '--train', action='store_true', 
                         help='Run training. Ignore to test.')
    
    #Training arguments
    tr_args = parser.add_argument_group('train', 'Training arguments')
    
    tr_args.add_argument('-W', '--width', action='store', nargs='?', 
                         default=286, type=restricted_int, 
                         help=('Input width for training. Image will be '
                               'resized to input width, then randomly '
                               'cropped to width 256. Default: 286'))
    tr_args.add_argument('-H', '--height', action='store', nargs='?', 
                         default=286, type=restricted_int, 
                         help=('Input height for training. Image will be '
                               'resized to input height, then randomly '
                               'cropped to height 256. Default: 286'))
    tr_args.add_argument('-x', '--lambda1', action='store', nargs='?', 
                         default=10.0, type=positive_float, 
                         help=('Weight for forward cyclic loss (A -> B -> A). '
                               'Default: 10.0'))
    tr_args.add_argument('-y', '--lambda2', action='store', nargs='?', 
                         default=10.0, type=positive_float, 
                         help=('Weight for backward cyclic loss (B -> A -> B). '
                               'Default: 10.0'))
    tr_args.add_argument('-p', '--poolsize', action='store', nargs='?', 
                         default=50, type=positive_int, 
                         help=('Number of fake images to store for calculating '
                               'loss. Deault: 50'))
    tr_args.add_argument('-l', '--lr', action='store', nargs='?', 
                         default=0.0002, type=positive_float, 
                         help=('Base learning rate. Default: 0.0002'))
    tr_args.add_argument('-e', '--epochs', action='store', nargs='?', 
                         default=200, type=positive_int, 
                         help='Max epochs. Default: 200')
    tr_args.add_argument('-s', '--save', action='store', nargs='?', default=0, 
                         type=positive_or_zero_int, 
                         help=('Number of training images to save for random '
                               'subjects. Default: 0'))
    tr_args.add_argument('-b', '--batch', action='store', nargs='?', default=1, 
                         type=positive_int, help='Batch size. Default: 1')
    tr_args.add_argument('-r', '--restore', action='store_true', 
                         help=('Restore previous checkpoint. Only used during '
                               'training. Ignore for new training run.'))
    tr_args.add_argument('-d', '--dir', action='store', nargs='?', type=str, 
                         help=('Name of folder containing checkpoints in '
                               'train/name/. Required for testing or when '
                               'restoring checkpoint during training.'))
    
    #Testing arguments
    te_args = parser.add_argument_group('test', 'Testing arguments')
    te_args.add_argument('-i', '--ids1', action='store', nargs='*', type=str, 
                         help='Subject IDs from group A to test.')
    te_args.add_argument('-j', '--ids2', action='store', nargs='*', type=str, 
                         help='Subject IDs from group B to test.')
    
    return parser.parse_args()

def main():
    args = parse()
    
    is_train = args.train
    restore_ckpt = args.restore
    timestamp = args.dir
    ids_a = args.ids1
    ids_b = args.ids2
    
    if is_train and restore_ckpt and timestamp is None:
        parser.error(('To resume training from previous checkpoint, training '
                      'output folder must be provided. See -d.'))
    if not is_train and timestamp is None:
        parser.error(('To run testing, training output folder containing '
                      'checkpoints must be provided. See -d.'))
    if not is_train and ids_a is None and ids_b is None:
        parser.error(('No testing subjects selected. See -i and -j.'))
    
    #Image dimensions
    input_w = args.width
    input_h = args.height
    
    #Min/max voxel intensities
    min_vox = args.min
    max_vox = args.max
    
    #Dataset name
    name = args.name
    
    #Training parameters
    lambda_a = args.lambda1
    lambda_b = args.lambda2
    pool_size = args.poolsize
    base_lr = args.lr
    max_step = args.epochs
    n_save = args.save
    batch_size = args.batch
    
    #File paths
    train_dir = os.path.join('train/', name)
    test_dir = os.path.join('test/', name)
    input_dir = os.path.join('datasets/', name)
    
    cyclegan = CycleGAN(input_w, input_h, min_vox, max_vox, name, lambda_a, 
                        lambda_b, pool_size, base_lr, max_step, n_save, 
                        batch_size, is_train, restore_ckpt, train_dir, 
                        test_dir, input_dir, timestamp, ids_a, ids_b)
    if is_train:
        cyclegan.train()
    else:
        cyclegan.test()

if __name__ == '__main__':
    main()

