#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
path_tutorial=${DIR}'/../tutorial/'
path_temp=$(dirname $(mktemp -u))/

FILES=$(find ${path_tutorial} -type f -name '*.ipynb')
FILECNT=$(find ${path_tutorial} -maxdepth 1 -name "*.ipynb" | wc -l)
report_file='report.txt'
cnt=0
fail=0
declare -a failedtest
for filename in $FILES; do
    fname=$(basename -- "$filename")
    extension="${fname##*.}"
    fname="${fname%.*}"
    echo ${path_temp}${fname}
    jupyter nbconvert --to script ${filename} --output ${path_temp}${fname}
    if ipython ${path_temp}${fname}'.py'; then
        echo "$fname test passed" >> $report_file
        ((cnt=cnt+1))
    else
        echo "$fname test failed" >> $report_file
        fail=1
    fi
done

rm ${path_temp}/*.py

echo $cnt 'tests passed out of' ${FILECNT} 'tutorial tests' >> $report_file

if [ $fail -eq 0 ] ; then
    exit 0
else
    exit 1
fi