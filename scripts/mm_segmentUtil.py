#TO DO: add descriptions to functions

import os
import logging
import sys
import json

#check_image_exists (DESCRIPTION)
def check_image_exists(image_path):
    if not os.path.isfile(image_path):
        logging.error(f"Image file '{image_path}' does not exist or is not a file.")
        sys.exit(1)
    if not os.access(image_path, os.R_OK):
        logging.error(f"Image file '{image_path}' is not readable.")
        sys.exit(1)

def get_model_and_config_paths(region, specified_model=None):
    models_base_dir = os.path.join("musclemap", "models", region)
    if specified_model:
        model_path = os.path.join(models_base_dir, specified_model)
        config_path = os.path.join(models_base_dir, f"{specified_model}.json")
        if not os.path.isfile(model_path):
            logging.error(f"Specified model '{specified_model}' does not exist for region '{region}'.")
            sys.exit(1)
        if not os.path.isfile(config_path):
            logging.error(f"Config file for model '{specified_model}' does not exist.")
            sys.exit(1)
    else:
        model_path = os.path.join(models_base_dir, "best_model.pth")
        config_path = os.path.join(models_base_dir, "config.json")
        if not os.path.isfile(model_path):
            logging.error(f"Best model for region '{region}' does not exist.")
            sys.exit(1)
        if not os.path.isfile(config_path):
            logging.error(f"Config file for the best model in region '{region}' does not exist.")
            sys.exit(1)
    
    return model_path, config_path


def load_model_config(config_path):
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        logging.error(f"Error: The configuration file '{config_path}' was not found.")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        logging.error(f"Error parsing the configuration file: {exc}")
        sys.exit(1)


def validate_arguments(args):
    if not args.image:
        logging.error("Error: The input image (-i) argument is required.")
        sys.exit(1)
    if args.image and not isinstance(args.image, str):
        logging.error("Error: The input image (-i) argument must be a string.")
        sys.exit(1)    
    
    if not args.region:
        logging.error("Error: The body region (-r) argument is required.")
        sys.exit(1)
    if args.region and not isinstance(args.region, str):
        logging.error("Error: The body region (-r) argument must be a string.")
        sys.exit(1)  

    # Optional Argument input=type string validation
    if args.model and not isinstance(args.model, str):
        logging.error("Error: The model (-m) argument must be a string.")
        sys.exit(1)
    
    if args.output_file_name and not isinstance(args.output_file_name, str):
        logging.error("Error: The output file name (-o) argument must be a string.")
        sys.exit(1)