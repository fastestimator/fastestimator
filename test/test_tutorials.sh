#!/bin/bash
path_apphub='../apphub/'
path_tutorial='../tutorial/'
path_temp=$(dirname $(mktemp -u))

FILES=$(find ${path_tutorial} -type f -name '*.ipynb')
FILECNT=$(find ${path_tutorial} -maxdepth 1 -name "*.ipynb" | wc -l)
cnt=0

for filename in $FILES; do
    fname=$(basename -- "$filename")
    extension="${fname##*.}"
    fname="${fname%.*}"
    jupyter nbconvert --to script ${filename} --output ${path_temp}${fname}
    if ipython ${path_temp}${fname}'.py'; then
        ((cnt=cnt+1))
    else
        echo 'failed '${fname}
        echo $cnt 'tests passed out of' ${FILECNT} 'tests'
        exit 0
    fi
done
rm ${path_temp}/*.py
echo $cnt 'tests passed out of' ${FILECNT} 'tests'
exit 1