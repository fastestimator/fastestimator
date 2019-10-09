#!/bin/bash
path_apphub='../apphub/'
path_nlp=${path_apphub}'NLP/'
path_temp=$(dirname $(mktemp -u))

#training parameters to test the models
train_info='--epochs 2 --batch_size 2 --steps_per_epoch 10 --validation_steps 5 --model_dir None'
nb_train_info='-p epochs 2 -p batch_size 2 -p steps_per_epoch 10 -p validation_steps 5' #notebook parameters

cnt=0
# LSTM IMDB
echo -en '\n'
echo 'NLP'
echo 'LSTM IMDB'
echo -en '\n'

filepath=${path_nlp}'lstm_imdb/'
filename='lstm_imdb.py'

if fastestimator train ${filepath}${filename} ${train_info} --max_len 300; then
    ((cnt=cnt+1))
else
    exit 0
fi

nb_filename='lstm_imdb.ipynb'
nb_param_filename='/lstm_imdb_param.ipynb'
#inject a parameter cell
papermill --prepare-only ${filepath}${nb_filename}  ${path_temp}${nb_param_filename} ${nb_train_info} -p MAX_LEN 300
jupyter nbconvert --to script  ${path_temp}${nb_param_filename} --output 'lstm_imdb_param'

if ipython  ${path_temp}'/lstm_imdb_param.py'; then
    ((cnt=cnt+1))
else
    exit 0
fi
rm -rf /tmp/tmp*
rm  ${path_temp}${nb_param_filename}
rm  ${path_temp}'/lstm_imdb_param.py'
exit 1