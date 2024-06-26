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
    models_base_dir = os.path.join(os.path.dirname(__file__), "..", "models", region)
    
    if specified_model:
        model_path = os.path.join(models_base_dir, specified_model)
        config_path = os.path.splitext(model_path)[0] + ".json"
        if not os.path.isfile(model_path):
            logging.error(f"Specified model '{specified_model}' does not exist.")
            sys.exit(1)
        if not os.path.isfile(config_path):
            logging.error(f"Config file for model '{specified_model}' does not exist.")
            sys.exit(1)
    else:
        if not os.path.isdir(models_base_dir):
            logging.error(f"Region folder '{region}' does not exist.")
            sys.exit(1)
        
        # Assuming only one model file and one config file in each region folder
        model_path = None
        config_path = None

        for file in os.listdir(models_base_dir):
            if file.endswith(".pth"):
                model_path = os.path.join(models_base_dir, file)
            elif file.endswith(".json"):
                config_path = os.path.join(models_base_dir, file)

        if not model_path:
            logging.error(f"No model file found in region folder '{region}'.")
            sys.exit(1)
        if not config_path:
            logging.error(f"No config file found in region folder '{region}'.")
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
    if args.output_dir and not isinstance(args.output_dir, str):
        logging.error("Error: The output directory (-s) argument must be a string.")
        sys.exit(1)