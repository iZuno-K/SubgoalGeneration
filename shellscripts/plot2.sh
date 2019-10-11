#!/usr/bin/env bash

# 2018/10/26
#base=/home/isi/karino/master/SubgoalGeneration/data/data7/sac/
#add=(MountainCarContinuousOneTurn-v0/1025/* MountainCarContinuous-v0/1025/*)
#for p in ${add[@]}; do
# for file in ${base}${p}; do
#   echo ${file}
##    python ../misc/plotter/maze_plotter.py --root-dir ${file} &
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

#dirs="/mnt/ISINAS1/karino/SubgoalGeneration/data/improve_exploration/sac/HalfCheetah-v2/0716/^/mnt/ISINAS1/karino/SubgoalGeneration/data/improve_exploration/sac/GMMPolicy/HalfCheetah-v2/0715"
dirs="/mnt/ISINAS1/karino/SubgoalGeneration/data/improve_exploration/sac/Walker2d-v2/0716/^/mnt/ISINAS1/karino/SubgoalGeneration/data/improve_exploration/sac/GMMPolicy/Walker2d-v2/0715"
dirs="/mnt/ISINAS1/karino/SubgoalGeneration/data/improve_exploration/sac/MountainCarContinuousOneTurn-v0/0716/^/mnt/ISINAS1/karino/SubgoalGeneration/data/improve_exploration/sac/GMMPolicy/MountainCarContinuousOneTurn-v0/0715"

dirs="/home/karino/tmp/home/karino/tmp_logfiles/improve_exploration/sac_logonly/HalfCheetah-v2/0716/^/home/karino/tmp/tmp/karino/kanck/improve_exploration/sac/GMMPolicy_logonly/HalfCheetah-v2/0716"

env=MountainCarContinuousOneTurn-v0
#env=HalfCheetah-v2
env=Walker2d-v2
e=0.05

#dirs=/tmp/karino/kanck/improve_exploration/sac/${env}/0717/^/home/karino/tmp/tmp/karino/kanck/improve_exploration/sac/GMMPolicy_logonly/${env}/0716
#dirs=~/Programs/master/SubgoalGeneration/data/tmp_karino/multiple_knack/improve_exploration/sac_logonly/${env}/0718/^/home/karino/tmp/tmp/karino/kanck/improve_exploration/sac/GMMPolicy_logonly/${env}/0716
#dirs=~/Programs/master/SubgoalGeneration/data/tmp_karino/multiple_knack/improve_exploration/sac_logonly/${env}/0718/^/home/karino/Programs/master/SubgoalGeneration/data/tmp_karino/karino/kanck/improve_exploration/sac/GMMPolicy_logonly/${env}/0719/
START=/mnt/ISINAS1/karino/SubgoalGeneration/data
start=/mnt/ISINAS1/karino/SubgoalGeneration/ParameterSearch

#dirs=${START}/improve_exploration/sac/GMMPolicy/${env}/0719/^${START}/EExploitation/e0.05${env}/^${START}/EExploitation/e0.1${env}/^${START}/EExploitation/e0.2${env}/^${START}/EExploitation/e0.3${env}/^${START}/MultipleKnack0.95/${env}/
#labels="DefaultExploration^EExploitation0.05^EExploitation0.1^EExploitation0.2^EExploitation0.3^KnackExploration
dirs=${START}/improve_exploration/sac/GMMPolicy/${env}/0719/^${START}/EExploitation/e0.3${env}/^${START}/EExploitation/e0.35${env}/^${START}/EExploitation/e0.4${env}/^${START}/MultipleKnack0.95/${env}/^${START}/SingedVariance0.95/${env}/^${START}/NegativeSingedVariance0.95/${env}/^${start}/SmallVariance0.95/${env}/^${start}/kurtosis-signed_variance0.95/${env}/^${start}/kurtosis-negative_signed_variance0.95/${env}/^${start}/kurtosis-small_variance0.95/${env}/
labels="DefaultExploration^EExploitation0.3^EExploitation0.35^EExploitation0.4^KnackExploration^SignedVariance^NegativeSignedVariance^SmallVariance^kurto-signed_var^kurto-negative_signed_var^kurto-var"


dirs=${start}/GMMPolicy/${env}/^${start}/Ssavearray/Knack-exploration0.95/${env}/^${start}/SmallVariance0.95/${env}/^${start}/kurtosis-signed_variance0.95/${env}/^${start}/kurtosis-negative_signed_variance0.95/${env}/^${start}/kurtosis-small_variance0.95/${env}/
labels="Default^Knack^SmallVariance^kurto-signed_var^kurto-negative_signed_var^kurto-var"
#dirs=${START}/improve_exploration/sac/GMMPolicy/${env}/0719/^${START}/MultipleKnack0.95/${env}/
#labels="Default^KnackExploration"
mode="exploitation"

echo "--root-dirs ${dirs} --labels ${labels} --mode ${mode} --smooth 50"
python ../misc/plotter/return_plotter.py --root-dirs ${dirs} --labels ${labels} --mode ${mode} --smooth 50 --plot_mode iqr
