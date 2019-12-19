#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
path_apphub=${DIR}'/../apphub/'
path_test_scripts=${DIR}'/test_apphub_scripts3'
test_apphub_scripts=$(ls $path_test_scripts)
tmpdir=$(dirname $(mktemp -u))
fail=0
report_file="report.txt"
stderr_dir="stderr"

# clear and create stderr folder 
if [ -d $stderr_dir ]; then
    rm -rf $stderr_dir
fi
mkdir $stderr_dir

# clear report file
if [ -f $report_file ]; then
    rm $report_file
fi

# Image Classification examples test
for file in $test_apphub_scripts; do
    echo "start running testing script "$file
    bash $path_test_scripts'/'$file
    if [ $? -eq 0 ] ; then
        echo ${file%.*} 'test passed' >> $report_file

    else 
        echo ${file%.*} 'test failed' >> $report_file
        fail=1
    fi
done
cat $report_file

grep -r "E tensorflow" stderr/

# #Tutorials examples test
# bash ${DIR}'/test_tutorials.sh'
# if [ $? -ne 0 ] ; then
#     fail=1
# fi

# cat $report_file
# rm $report_file

# if [ $fail -eq 0 ] ; then
#     exit 0
# else
#     exit 1
# fi
