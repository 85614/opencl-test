conda create -n tf --clone rcnn

conda remove -n rcnn --all

conda info -e

source ~/.bashrc

python3 -m pip install -e ./python/