#!/bin/bash
# Run from root folder with: bash bash/setup_conda.sh

# This line is needed for enabling conda env activation
source ~/.bashrc

# check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "The 'conda' command could not be found. Exiting..."
    exit
fi

# Configure conda env
read -rp "Enter environment name [{{cookiecutter.project_slug}}]: " env_name
env_name=${env_name:-{{cookiecutter.project_slug}}}

read -rp "Enter python version [3.10]:" python_version
python_version=${python_version:-3.10}

# Create conda env
conda create -y -n "$env_name" python="$python_version"
conda activate "$env_name"

conda install -c conda-forge gdal 

echo "\n"
echo "To activate this environment, use:"
echo "conda activate $env_name"
