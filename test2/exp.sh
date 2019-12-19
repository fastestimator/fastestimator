#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
FILES=$(ls $DIR'/test_apphub_scripts')

# for f in $FILES; do
#     file=${f%.*}
#     echo $file
#     echo ","
# done



if bash exp3.sh; then 
    echo "success"
fi
