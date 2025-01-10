import os
from pathlib import Path

# create project structure
project_name = "AUTOFORECAST"
list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_training.py",
    f"src/{project_name}/components/model_evaluation.py",
    f"src/{project_name}/components/forecasting.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/pipeline.py",
    f"src/{project_name}/pipeline/stage_01_data_transormation.py",
    f"src/{project_name}/pipeline/stage_02_model_training.py",
    f"src/{project_name}/pipeline/stage_03_model_evaluation.py",
    f"src/{project_name}/pipeline/stage_04_forecasting.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "artifact/data",
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
        print(f"Created directory : {filedir} for the file {filename}")
    else:
        print.info(f"Directoy ({filedir}) already exists.")
    if not os.path.exists(filepath):
        with open(filepath, "w") as f:
            print.info(f"Created empty file : {filepath}")

    else:
        print.info(f"File ({filepath}) already exists.")
