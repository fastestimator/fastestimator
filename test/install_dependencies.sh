JENKINS_DIR=$WORKSPACE/..
PATH=$JENKINS_DIR/venv/bin:/usr/local/bin:$PATH
echo $JENKINS_DIR
echo $PATH
if [ ! -d "$JENKINS_DIR/venv"  ]; then
    virtualenv -p python3.6 $JENKINS_DIR/venv
fi
. $JENKINS_DIR/venv/bin/activate

pip3 install tensorflow==1.12.0 pytest pytest-cov numpy nibabel pydicom
pip3 install --upgrade setuptools
pip3 install -e .
