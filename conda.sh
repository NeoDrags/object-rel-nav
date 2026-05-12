conda create -n nav python=3.10
conda activate nav
conda install mamba -c conda-forge
mamba install habitat-sim=0.2.4 withbullet cmake=3.27 -c aihabitat -c conda-forge
