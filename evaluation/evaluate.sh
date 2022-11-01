#!/bin/sh

# read cmd arguments
while getopts m:t:p:d:e:b:s:c: flag
do
    case "${flag}" in
        m) models=${OPTARG};;
        t) tasks=${OPTARG};;
        p) path=${OPTARG};;
        d) device=${OPTARG};;
        e) epochs=${OPTARG};;
        b) batch=${OPTARG};;
        s) seeds=${OPTARG};;
        c) city=${OPTARG};;
    esac
done

export IFS=","
i=0
for model in $models; do
    for task in $tasks; do
        for seed in $seeds; do
            echo "Start Evaluation Number $i | model: $model | task: $task | seed: $seed"
            python -u evaluate_models.py -m $model -t $task -p $path -d $device -e $epochs -b $batch -se $seed -c $city
            i=$(($i+1))
        done
    done
done

# fuse results into task specific result files
python fuse_results.py -p $path
rm -r $path*/