import os

import yaml
from django.conf import settings


def load_yaml_file(file_name):
    # Optional: Only allow .yaml or .yml files to prevent unwanted access.
    if not (file_name.endswith('.yaml') or file_name.endswith('.yml')):
        raise ValueError("Invalid file extension. Please provide a '.yaml' or '.yml' file.")

    # Build the absolute path to the YAML file
    file_path = os.path.join(
        settings.BASE_DIR,
        'prototype',
        'utils',
        'ai_prompts',
        file_name
    )

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    # Open and load the YAML file safely
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    return data