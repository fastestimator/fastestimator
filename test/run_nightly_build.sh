#!/bin/bash

full_path=$(realpath $0)
dir_path=$(dirname $full_path)

declare -A exectime
declare -A result

# run apphub
for file in $(find $dir_path/apphub_scripts -type f); do
    if [[ $file == *.sh ]]; then
        if [[ $file == *template* ]]; then
            continue
        fi
        echo $file
        start=`date +%s`
        bash $file
        result[$file]=$?
        end=`date +%s`

        # clean GPU memory
        if ls /dev/nvidia* 1> /dev/null 2>&1; then
            for i in $(sudo lsof /dev/nvidia* | awk 'FNR>1 {print $2}' | sort -u); do
                sudo kill -9 $i;
            done
        fi

        exectime[$file]=$((end-start))
        if [ ! ${result[$file]} -eq 0 ]; then
            echo "---------------- error log of $file-------------------"
            cat "${file/'.sh'/'_stderr.txt'}"
            echo "------------------------------------------------------"
        fi
    fi
done

rm -rf $dir_path"/tutorial"
cp -r $(realpath $dir_path/../tutorial) $dir_path

# run tutorial
for nb_in in $(find $dir_path/tutorial -type f); do
    if [[ $nb_in == *.ipynb ]]; then
        echo $nb_in
        nb_out=${nb_in/'.ipynb'/'_out.ipynb'}
        current_dir=$(dirname $nb_in)
        stderr_file=${nb_in/'.ipynb'/'_stderr.txt'}
        start=`date +%s`
        papermill $nb_in $nb_out 2>> $stderr_file -k nightly_build --cwd $current_dir
        result[$nb_in]=$?
        end=`date +%s`

        # clean GPU memory
        if ls /dev/nvidia* 1> /dev/null 2>&1; then
            for i in $(sudo lsof /dev/nvidia* | awk 'FNR>1 {print $2}' | sort -u); do
                sudo kill -9 $i;
            done
        fi

        exectime[$nb_in]=$((end-start))
        if [ ! ${result[$nb_in]} -eq 0 ]; then
            echo "---------------- error log of $nb_in-------------------"
            cat "${nb_in/'.sh'/'_stderr.txt'}"
            echo "-------------------------------------------------------"
        fi

    fi
done

# print report
echo "------------------------ report ------------------------"
for key in ${!exectime[@]}; do
    if [ ${result[$key]} -eq 0 ]; then
        echo "$key: pass, (spend ${exectime[$key]} seconds)"
    else
        echo "$key: fail, (spend ${exectime[$key]} seconds)"
    fi
done

# print fail list
echo "---------------------- fail list -----------------------"
is_fail=0
for key in ${!exectime[@]}; do
    if [ ! ${result[$key]} -eq 0 ]; then
        echo "$key"
        is_fail=1
    fi
done

if [ $is_fail -eq 0 ]; then
    echo "all tests passed"
    exit 0
else
    exit 1
fi
