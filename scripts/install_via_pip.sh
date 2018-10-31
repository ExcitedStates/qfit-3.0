#! /bin/bash
pip install numpy scipy osqp

cd ../
git clone git@github.com:oxfordcontrol/miosqp.git
cd miosqp
python setup.py install

cd ../qfit-3.0
python setup.py install
