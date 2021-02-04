echo rm input output
rm input output
echo in mxnetcpu
conda activate mxnetcpu
python ./test.py
echo in mxnetopencl
conda activate mxnetopencl
python ./test.py