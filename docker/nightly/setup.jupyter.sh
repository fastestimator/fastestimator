#!/bin/bash
jupyter serverextension enable --py jupyter_http_over_ws

mkdir /.local
chmod 755 /.local
apt-get update
apt-get install -y --no-install-recommends wget git

apt-get autoremove -y
apt-get remove -y wget

python3 -m ipykernel.kernelspec
