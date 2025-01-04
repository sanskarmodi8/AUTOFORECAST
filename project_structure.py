import logging
import os
from pathlib import Path

# set up logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] : %(message)s")

# create project structure
project_name = "AUTOFORECAST"
list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_analysis.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_training.py",
    f"src/{project_name}/components/model_evaluation.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/prediction.py",
    f"src/{project_name}/pipeline/stage_01_data_analysis.py",
    f"src/{project_name}/pipeline/stage_02_data_transormation.py",
    f"src/{project_name}/pipeline/stage_03_model_training.py",
    f"src/{project_name}/pipeline/stage_04_model_evaluation.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "main.py",
    "requirements.txt",
    "app.py",
    "Dockerfile",
    ".env",
    "pyproject.toml",
    "format.sh",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "" and not os.path.exists(filedir):
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory : {filedir} for the file {filename}")
    else:
        logging.info(f"Directoy ({filedir}) already exists.")
    if not os.path.exists(filepath):
        with open(filepath, "w") as f:
            logging.info(f"Created empty file : {filepath}")
        
    else:
        logging.info(f"File ({filepath}) already exists.")