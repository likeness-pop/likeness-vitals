#!/bin/bash
# check for NVIDIA GPU
if lspci | grep -i nvidia; then
  echo "NVIDIA GPUs found. Installing environment ``py312_likeness_gh_cuda``."
  env="_cuda"
else
  echo "CPU only. Installing environment ``py312_likeness_gh``."
  env=""
fi

# install and activate base environment
conda env create -f $"envs/py312_likeness_gh$env.yml"
conda_path=$(which conda)
conda_activate_path=$(echo "${conda_path/"/conda"/"/activate"}")
source activate $conda_activate_path
conda activate $"py312_likeness_gh$env"

# install pmedm_legacy solver
# install R packages from source
R --silent -e "remotes::install_bitbucket('jovtc/pmedmrcpp')"
R --silent -e "IRkernel::installspec()"
R --silent -e "remotes::install_git('https://github.com/likeness-pop/pmedmrcpp-pal.git@develop')"
pip install git+https://github.com/likeness-pop/pmedm-legacy.git@develop --no-deps --no-cache-dir
