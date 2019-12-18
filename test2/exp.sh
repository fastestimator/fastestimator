#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
FILES=$(ls $DIR'/test_apphub_scripts')

# for f in $FILES; do
#     file=${f%.*}
#     echo $file
#     echo ","
# done

a=0
if [ $a -eq  0 ]; then
    echo "yes"
else
    echo "no"
fi