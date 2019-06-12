#!/bin/bash

#get the direcotry of tests.sh
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo -en '\n\n'
echo +++++++++++++++++ Creating MNIST data +++++++++++++++++++++++
python $DIR/create_mnist_data.py

echo -en '\n\n'
echo +++++++++++++++++ Integration testing +++++++++++++++++++++++
echo Setup parameters to test
echo -en '\n\n'
cnt=0
data_type="numpy csv tfrecords"
metrics="custom regular"
filter="custom regular"
lr_schedule="custom regular"


echo -en '\n\n'
echo -----------------------------
echo 1. Test with just entry point
echo -----------------------------
if fastestimator train --entry_point $DIR/test.py; then
    ((cnt=cnt+1))
else
    echo Testing failed on test with just entry point
fi

echo -en '\n\n'
echo ------------------
echo 2. Test data types
echo ------------------
for t in $data_type
do
    if fastestimator train --entry_point $DIR/test.py --input tfrecords --data_type $t --compression GZIP; then
    ((cnt=cnt+1))
    else
        echo Testing failed on test with $t
    fi
done

echo -en '\n\n'
echo ------------------------------
echo 3. Test with validation splits
echo ------------------------------
for t in $data_type
do
    if fastestimator train --entry_point $DIR/test.py --input tfrecords --data_type $t --val_split 0.2 --compression ZLIB; then
    ((cnt=cnt+1))
    else
        echo Testing failed on test with validation split
    fi
done

echo -en '\n\n'
echo ---------------
echo 4. Test losses
echo ---------------
if fastestimator train --entry_point $DIR/test.py --input tfrecords --data_type tfrecords --loss custom; then
((cnt=cnt+1))
else
    echo Testing failed on test with loss as custom
fi

echo -en '\n\n'
echo ----------------
echo 5. Test metrics
echo ----------------
for t in $metrics
do
    if fastestimator train --entry_point $DIR/test.py --input tfrecords --data_type tfrecords --metrics $t; then
    ((cnt=cnt+1))
    else
        echo Testing failed on test with metrics as $t
    fi
done

echo -en '\n\n'
echo -----------------
echo 6. Test optimizer
echo -----------------
if fastestimator train --entry_point $DIR/test.py --input tfrecords --data_type tfrecords --optimizer custom; then
((cnt=cnt+1))
else
    echo Testing failed on test with optimizer as custom
fi

echo -en '\n\n'
echo -----------------------
echo 7. Test model directory
echo -----------------------
if fastestimator train --entry_point $DIR/test.py --input tfrecords --data_type tfrecords --model_dir model; then
((cnt=cnt+1))
else
    echo Testing failed on test with model directory as model/
fi

echo -en '\n\n'
echo --------------
echo 8. Test filter
echo --------------
for t in $filter
do
    if fastestimator train --entry_point $DIR/test.py --input tfrecords --data_type tfrecords --filter $t; then
    ((cnt=cnt+1))
    else
        echo Testing failed on test with filter as $t
    fi
done

echo -en '\n\n'
echo -----------------------------------------------------
echo 9. Test steps per epoch and validation steps per epoch
echo -----------------------------------------------------
if fastestimator train --entry_point $DIR/test.py --input tfrecords --data_type tfrecords --steps_per_epoch None --validation_steps None; then
((cnt=cnt+1))
else
    echo Testing failed on test with optimizer as $t
fi


echo -en '\n\n'
echo ---------------------
echo 10. Test LR scheduler
echo ---------------------
for t in $lr_schedule
do
    if fastestimator train --entry_point $DIR/test.py --input tfrecords --data_type tfrecords --lr_schedule $t; then
    ((cnt=cnt+1))
    else
        echo Testing failed on test with lr schedule as $t
    fi
done


echo -en '\n\n'
echo --------------------------------------------------------------
echo 11. Test LR scheduler with decrease method and number of cycles
echo --------------------------------------------------------------
if fastestimator train --entry_point $DIR/test.py --input tfrecords --data_type tfrecords --lr_schedule cosine --decrease_method linear --num_cycle 1; then
((cnt=cnt+1))
else
    echo Testing failed on test with lr schedule as cosine and decrease method as linear and number of cycles as 1
fi



echo -en '\n\n'
echo ------------------------
echo 12. Test custom pipeline
echo ------------------------
if fastestimator train --entry_point $DIR/test.py --input tfrecords --data_type tfrecords --pipeline custom; then
((cnt=cnt+1))
else
    echo Testing failed on test with custom pipeline as $t
fi


echo -en '\n\n'
echo ------------------------------------
echo 13. Test preprocess and augmentation
echo ------------------------------------
if fastestimator train --entry_point $DIR/test.py --input tfrecords --data_type tfrecords --preprocess custom --augment custom; then
((cnt=cnt+1))
else
    echo Testing failed on test with lr schedule as $t
fi



echo -en '\n\n'
echo ------------------------------------
echo 14. Test reduce LR on plateau
echo ------------------------------------
if fastestimator train --entry_point $DIR/test.py --input tfrecords --data_type tfrecords --reduce_lr True; then
((cnt=cnt+1))
else
    echo Testing failed on test with lr schedule as $t
fi

echo $cnt tests passed out of 21 tests

rm -rf mnist/
rm -rf tfrecords/
rm annotation*.csv
rm ~/.keras/datasets/mnist.npz
