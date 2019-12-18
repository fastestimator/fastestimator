#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
path_apphub=${DIR}'/../apphub/'
path_test_scripts=${DIR}'/test_apphub_scripts'
test_apphub_scripts=$(ls $path_test_scripts)

tmpdir=$(dirname $(mktemp -u))

fail=0
report_file="report.txt"

# Image Classification examples test
for file in $test_apphub_scripts; do 
    bash $path_test_scripts'/'$file
    if [ $? -eq 0 ] ; then
        echo ${file%.*} 'test passed' >> $report_file

    else 
        echo ${file%.*} 'test failed' >> $report_file
        fail=1
    fi
done

#Tutorials examples test
bash ${DIR}'/test_tutorials.sh'
if [ $? -ne 0 ] ; then
    fail=1
fi

cat $report_file
rm $report_file

if [ $fail -eq 0 ] ; then
    exit 0
else
    exit 1
fi
