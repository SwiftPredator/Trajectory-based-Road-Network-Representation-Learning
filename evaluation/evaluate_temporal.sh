#!/bin/sh

# read cmd arguments
while getopts m:t:p:d:e:b:l: flag
do
    case "${flag}" in
        m) models=${OPTARG};;
        t) tasks=${OPTARG};;
        p) path=${OPTARG};;
        d) device=${OPTARG};;
        e) epochs=${OPTARG};;
        b) batch=${OPTARG};;
        l) lr=${OPTARG};;
    esac
done

export IFS=","
i=0
for model in $models; do
    for task in $tasks; do
        echo "Start Temporal Evaluation Number $i | model: $model | task: $task"
        python evaluate_temporal.py -m $model -t $task -p $path -d $device -e $epochs -lr $lr -b $batch
        i=$(($i+1))
    done
done

# fuse results into task specific result files
python fuse_results.py -p $path
#rm -r $path*/