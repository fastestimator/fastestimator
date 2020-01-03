DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
path_tutorial=${DIR}'/../tutorial/'
path_temp=$(dirname $(mktemp -u))/

filename="${path_tutorial}t02_using_data_in_disk.ipynb"
# filename="${path_tutorial}t11_interpretation.ipynb"
# filename="${path_tutorial}t07_expand_data_dimension"

fname=$(basename -- "$filename")
extension="${fname##*.}"
fname="${fname%.*}"
echo ${path_temp}${fname}
jupyter nbconvert --to script ${filename} --output ${path_temp}${fname}
if ipython ${path_temp}${fname}'.py'; then
    echo "$fname test passed" 
else
    echo "$fname test failed"
    fail=1
fi