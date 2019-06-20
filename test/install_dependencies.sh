PATH=$WORKSPACE/venv/bin:/usr/local/bin:$PATH
echo $WROKSPACE
if [ ! -d "venv"  ]; then
    virtualenv -p python3.6 venv
fi
. venv/bin/activate

pip3 install tensorflow==1.12.0 pytest numpy nibabel pydicom
pip3 install -e .
