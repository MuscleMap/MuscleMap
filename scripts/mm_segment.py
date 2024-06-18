#!/usr/bin/env python
# -*- coding: utf-8

# For usage, type: python mm_segment.py -h

# Authors: Richard Yin, Eddo Wesselink, and Kenneth Weber

import argparse
import os

def get_parser():
    parser = argparse.ArgumentParser(
        description="Segment an input image according to the specified deep learning model.")
    
    #Required arguments
    required = parser.add_argument_group("Required")
    required.add_argument("-i", required=True, type=str,
                        help="Image to segment. Can be multiple images separated with spaces.")
    required.add_argument("-r", required=True, type=str,
                        help="Body region of input image.")
    
    #Optional arguments
    optional = parser.add_argument_group("Optional")
    optional.add_argument("-m", default=None, required=False, type=str,
                        help="Option to specifiy another model.")
    optional.add_argument("-o", default=None, required=False, type=str,
                        help="Output file name. By default, dseg suffix will be added, and the output extension will be .nii.gz.")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    #Check that image exists and is readable

    #Check that body region and corresponding model are available

    #Load model and model parameters

    #Run inference on the image and images.


    """Optional"""
    #Allow list of filenames separated by spaces
    #Make sure relative and absolute paths work
    

if __name__ == "__main__":
    main()
    