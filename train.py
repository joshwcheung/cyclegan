import argparse
import os

from models.cyclegan import CycleGAN
from util.parser import training_parser

def main():
    args = training_parser().parse_args()
    
    #Arguments
    min_vox, max_vox = args.min, args.max
    name = args.name
    axis = args.axis
    input_w, input_h = args.width, args.height
    lambda_a, lambda_b = args.lambda1, args.lambda2
    pool_size = args.pool
    base_lr = args.lr
    max_step = args.epochs
    n_save = args.save
    batch_size = args.batch
    restore = args.restore
    restore_ckpt = True if restore else False
    
    #File paths
    train_dir = os.path.join('train/', name)
    input_dir = os.path.join('datasets/', name)
    
    cyclegan = CycleGAN(input_w, input_h, min_vox, max_vox, name, lambda_a, 
                        lambda_b, pool_size, base_lr, max_step, n_save, 
                        batch_size, axis, True, restore_ckpt, train_dir, 
                        None, input_dir, restore, None, None)
    cyclegan.train()

if __name__ == '__main__':
    main()

