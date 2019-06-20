PATH=$WORKSPACE/venv/bin:/usr/local/bin:$PATH
echo $WROKSPACE
if [ ! -d "venv"  ]; then
    virtualenv -p python3.6 venv
fi
. venv/bin/activate

pip3 install pytest numpy nibabel pydicom horovod
pip3 install -e .
