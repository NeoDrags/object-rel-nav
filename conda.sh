conda create -n nav python=3.9
conda activate nav
conda install mamba -c conda-forge
mamba install habitat-sim=0.3.1 withbullet cmake=3.27 -c aihabitat -c conda-forge
