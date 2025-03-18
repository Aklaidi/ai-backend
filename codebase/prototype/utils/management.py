import os
from django.conf import settings

def get_csv_path(filename, extra_path: str | None = 'prototype/utils/training_data'):
    """
    Given the name of a CSV file, returns the absolute path
    in your Django project.
    """
    if not extra_path:
        base_dir = str(settings.BASE_DIR).replace("codebase", "")
    return os.path.join(settings.BASE_DIR, extra_path, filename) if extra_path else os.path.join(base_dir,filename)