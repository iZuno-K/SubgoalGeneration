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
file_path="/mnt/qnap_o1/karino/knack_experiments/ddpg/ContinuousSpaceMazeDoubleRevised/0925noL2/*"
for file in ${file_path}; do
    echo ${file}
    python ../misc/plotter.py --root-dir ${file} &
    python ../misc/plotter/experienced_states_plotter.py --root-dir ${file} &
    sleep 30s
done
