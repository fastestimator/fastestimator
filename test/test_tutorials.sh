#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
path_tutorial=${DIR}'/../tutorial/'
path_temp=$(dirname $(mktemp -u))

FILES=$(find ${path_tutorial} -type f -name '*.ipynb')
FILECNT=$(find ${path_tutorial} -maxdepth 1 -name "*.ipynb" | wc -l)

cnt=0
i=0
fail=0
declare -a failedtest
for filename in $FILES; do
    fname=$(basename -- "$filename")
    extension="${fname##*.}"
    fname="${fname%.*}"
    echo ${path_temp}${fname}
    jupyter nbconvert --to script ${filename} --output ${path_temp}${fname}
    if ipython ${path_temp}${fname}'.py'; then
        ((cnt=cnt+1))
    else
        failedtest[$i]=${fname}
        fail=1
    fi
    ((i=i+1))
done

rm ${path_temp}/*.py
for idx in "${failedtest[@]}"
do
   echo "$idx test failed"
done
echo $cnt 'tests passed out of' ${FILECNT} 'tutorial tests'

if [ $fail -eq 1 ] ; then
    exit 1
else
    exit 0
fi