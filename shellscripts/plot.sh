#!/usr/bin/env bash
#python ../misc/plotter.py --root-dir ../data5/DoubleContinuousSpaceMaze20_45_RB1e6_entropy0.0_epoch3000__Normalize_dynamicCoeff/seed1 &
#sleep 30s
#python ../misc/plotter.py --root-dir ../data5/DoubleContinuousSpaceMaze20_45_RB1e6_entropy1.0_epoch3000__Normalize_dynamicCoeff/seed1 &
#sleep 30s
#python ../misc/plotter.py --root-dir ../data5/EasierDoubleContinuousSpaceMaze20_45_RB1e6_entropy0.0_epoch3000__Normalize_dynamicCoeff/seed1 &
#sleep 30s
#python ../misc/plotter.py --root-dir ../data5/EasierDoubleContinuousSpaceMaze20_45_RB1e6_entropy1.0_epoch3000__Normalize_dynamicCoeff/seed1 &
#sleep 30s
#python ../misc/plotter.py --root-dir ../data5/SinglePathContinuousSpaceMaze20_45_RB1e6_entropy0.0_epoch3000__Normalize_dynamicCoeff/seed1 &
#sleep 30s
#python ../misc/plotter.py --root-dir ../data5/SinglePathContinuousSpaceMaze20_45_RB1e6_entropy1.0_epoch3000__Normalize_dynamicCoeff/seed1 &

#file_path="../data6/DoubleContinuousSpaceMaze20_45_RB1e6_entropy0.0_epoch3000__Normalize_uniform/*"
#for file in ${file_path}; do
#    echo ${file}
#    python ../misc/plotter.py --root-dir ${file} &
#    sleep 30s
#done

#file_path="../data6/MountainCarContinuous_RB1e6_entropy0.0_epoch1000__Normalize_uniform/*"
#for file in ${file_path}; do
#    echo ${file}
#    python ../misc/plotter.py --root-dir ${file} &
#    sleep 30s
#done

#file_paths=("/mnt/qnap_o1/karino/knack_experiments/ddpg/ContinuousSpaceMazeDoubleRevised/0920clip_norm/*" "/home/isi/karino/master/SubgoalGeneration/data/data7/sac")
# file_paths[@]
#file_path="/home/isi/karino/master/SubgoalGeneration/data/data7/sac/DoubleRevisedContinuousSpaceMaze20_45_RB1e6_entropy0.0_epoch3000__Normalize_uniform/0920/*"
#file_path="/mnt/qnap_o1/karino/knack_experiments/ddpg/ContinuousSpaceMazeDoubleRevised/0925noL2/*"
#file_path=/home/isi/karino/master/SubgoalGeneration/data/data7/sac/clip_norm/DoubleRevisedContinuousSpaceMaze20_45_RB1e6_entropy0.0_epoch3000__Normalize_uniform/1017/*
#for file in ${file_path}; do
#    echo ${file}
#    python ../misc/plotter.py --root-dir ${file} &
#    python ../misc/plotter/experienced_states_plotter.py --root-dir ${file} &
#    sleep 30s
#done

#2018/10/19 running average
#file_path=/home/isi/karino/master/SubgoalGeneration/data/data7/sac/clip_norm/DoubleRevisedContinuousSpaceMaze20_45_RB1e6_entropy0.0_epoch3000__Normalize_uniform/1017/*
#file_path="/home/isi/karino/master/SubgoalGeneration/data/data7/sac/DoubleRevisedContinuousSpaceMaze20_45_RB1e6_entropy0.0_epoch3000__Normalize_uniform/0920/*"
#file_path="/home/isi/karino/master/SubgoalGeneration/data/data7/sac/DoubleRevisedContinuousSpaceMaze20_45_RB1e6_entropy0.0_epoch3000__Normalize_uniform/0920/*"
#for file in ${file_path}; do
#    echo ${file}
#    python ../misc/plotter/running_average_plotter.py --root-dir ${file} --average-times 20 &
#    sleep 30s
#    python ../misc/plotter/running_average_plotter.py --root-dir ${file} --average-times 10 &
 #   sleep 30s
 #   python ../misc/plotter/running_average_plotter.py --root-dir ${file} --average-times 5 &
 #   sleep 30s
#done

# 2018/10/20
#base=/home/isi/karino/master/SubgoalGeneration/data/data7/sac/
#add=(ContinuousSpaceMazeDoubleRevisedDense/1020/* ContinuousSpaceMazeDoubleRevisedDenseTerminateDist/1020/* ContinuousSpaceMazeDoubleRevisedSparse/1020/* ContinuousSpaceMazeDoubleRevisedSparseTerminateDist/1020/*)
#for p in ${add[@]}; do
# for file in ${base}${p}; do
#   echo ${file}
#    python ../misc/plotter.py --root-dir ${file} &
#    sleep 30s
#    python ../misc/plotter/running_average_plotter.py --root-dir ${file} --average-times 20 &
#    sleep 30s
#    python ../misc/plotter/running_average_plotter.py --root-dir ${file} --average-times 10 &
#    sleep 30s
#    python ../misc/plotter/running_average_plotter.py --root-dir ${file} --average-times 5 &
# done
#done

# 2018/10/22
#dirs=/home/isi/karino/master/SubgoalGeneration/data/data7/sac/ContinuousSpaceMazeDoubleRevisedDense/1020clip_norm/*
#for dir in ${dirs}; do
#    echo ${dir}
#    python ../misc/plotter.py --root-dir ${dir} &
#    sleep 30s
#    python ../misc/plotter.py --root-dir ${dir} &
#    sleep 30s
#    python ../misc/plotter/running_average_plotter.py --root-dir ${dir} --average-times 20 &
#    sleep 30s
#    python ../misc/plotter/running_average_plotter.py --root-dir ${dir} --average-times 10 &
#    sleep 30s
#    python ../misc/plotter/running_average_plotter.py --root-dir ${dir} --average-times 5 &
#done

# 2018/10/22
#base=/home/isi/karino/master/SubgoalGeneration/data/data7/sac/
#add=(ContinuousSpaceMazeDoubleRevisedDense/1020/* ContinuousSpaceMazeDoubleRevisedDenseTerminateDist/1020/* ContinuousSpaceMazeDoubleRevisedSparse/1020/* ContinuousSpaceMazeDoubleRevisedSparseTerminateDist/1020/* ContinuousSpaceMazeDoubleRevisedDense/1020clip_norm/*)
#for p in ${add[@]}; do
# for file in ${base}${p}; do
#   echo ${file}
#    python ../misc/plotter.py --root-dir ${file} &
#    sleep 30s
#    python ../misc/plotter/experienced_states_plotter.py --root-dir ${file} &
#    sleep 30s
# done
#done

# 2018/10/23
base=/home/isi/karino/master/SubgoalGeneration/data/data7/sac/
#add=(ContinuousSpaceMazeDoubleRevisedDenseTerminateDist/1023/*  ContinuousSpaceMazeDoubleRevisedDenseTerminateDist/1023no_normalize/*)
#add=(ContinuousSpaceMazeDoubleRevisedDenseTerminateDist/1024bugfix/*  ContinuousSpaceMazeDoubleRevisedDenseTerminateDist/1024bugfix_no_normalize/*)
#add=(ContinuousSpaceMazeDoubleRevisedDenseTerminateDist/1024bugfixdone/*  ContinuousSpaceMazeDoubleRevisedDenseTerminateDist/1024bugfixdone_no_normalize/*)
add=(ContinuousSpaceMazeDoubleRevisedDense/1024bugfixdone_no_normalize/*)
for p in ${add[@]}; do
 for file in ${base}${p}; do
   echo ${file}
    python ../misc/plotter/running_average_plotter.py --root-dir ${file} --average-times 20 --exclude-fault 1 &
    sleep 30s
    python ../misc/plotter/running_average_plotter.py --root-dir ${file} --average-times 10 --exclude-fault 1 &
    sleep 30s
    python ../misc/plotter.py --root-dir ${file} &
    sleep 30s
    python ../misc/plotter/experienced_states_plotter.py --root-dir ${file} &
    sleep 30s
    python ../misc/plotter/running_average_plotter.py --root-dir ${file} --average-times 20 &
    sleep 30s
    python ../misc/plotter/running_average_plotter.py --root-dir ${file} --average-times 10 &
    sleep 30s
 done
done
