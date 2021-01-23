cd ~/mxnet_opencl/src
git pull
cd ~/mxnet_opencl
make -j48
cd ~

cd ~/opencl-test
git pull
python ./test.py
cd ~