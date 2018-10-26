#!/usr/bin/env bash

# 2018/10/26
#base=/home/isi/karino/master/SubgoalGeneration/data/data7/sac/
#add=(MountainCarContinuousOneTurn-v0/1025/* MountainCarContinuous-v0/1025/*)
#for p in ${add[@]}; do
# for file in ${base}${p}; do
#   echo ${file}
##    python ../misc/plotter.py --root-dir ${file} &
##    sleep 10s
##    python ../misc/plotter/experienced_states_plotter.py --root-dir ${file} --mode MountainCar &
##    sleep 10s
#    python ../misc/plotter/running_average_plotter.py --root-dir ${file} --average-times 20 --mode MountainCar &
#    sleep 10s
#    python ../misc/plotter/running_average_plotter.py --root-dir ${file} --average-times 10  --mode MountainCar &
#    sleep 10s
# done
#done

# 2018/10/26
dirs="/home/karino/mount/ContinuousSpaceMazeDoubleRevisedDense/1025KnackPControl^/home/karino/mount/ContinuousSpaceMazeDoubleRevisedDense/1025KnackExploitation^/home/karino/mount/ContinuousSpaceMazeDoubleRevisedDense/1025KnackExploration^/home/karino/mount/ContinuousSpaceMazeDoubleRevisedDense/1025completeBugfixno_normalize"

labels="KnackPControl^KnackExploitation^KnackExploration^Default"
echo ${dirs}
python ../misc/plotter/return_plotter.py --root-dirs ${dirs} --labels ${labels}