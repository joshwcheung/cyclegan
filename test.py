import argparse
import os

from models.cyclegan import CycleGAN
from util.parser import training_parser

def main():
    args = testing_parser().parse_args()
    
    #Arguments
    min_vox, max_vox = args.min, args.max
    name = args.name
    axis = args.axis
    ids_a, ids_b = args.ids1, args.ids2
    model = args.model
    
    #File paths
    train_dir = os.path.join('train/', name)
    test_dir = os.path.join('test/', name)
    input_dir = os.path.join('datasets/', name)
    
    cyclegan = CycleGAN(None, None, min_vox, max_vox, name, None, None, 0, 
                        None, None, 0, 1, axis, False, None, train_dir, 
                        test_dir, input_dir, model, ids_a, ids_b)
    cyclegan.test()

if __name__ == '__main__':
    main()

