#!/usr/bin/env bash

#env=MountainCarContinuousOneTurn-v0
#env=HalfCheetah-v2
env=Walker2d-v2
e=0.05

#START=/mnt/ISINAS1/karino/SubgoalGeneration/data
start=/mnt/ISINAS1/karino/SubgoalGeneration/ParameterSearch
#dirs=${START}/improve_exploration/sac/GMMPolicy/${env}/0719/^${START}/EExploitation/e0.3${env}/^${START}/EExploitation/e0.35${env}/^${START}/EExploitation/e0.4${env}/^${START}/MultipleKnack0.95/${env}/^${START}/SingedVariance0.95/${env}/^${START}/NegativeSingedVariance0.95/${env}/^${start}/SmallVariance0.95/${env}/^${start}/kurtosis-signed_variance0.95/${env}/^${start}/kurtosis-negative_signed_variance0.95/${env}/^${start}/kurtosis-small_variance0.95/${env}/
#labels="DefaultExploration^EExploitation0.3^EExploitation0.35^EExploitation0.4^KnackExploration^SignedVariance^NegativeSignedVariance^SmallVariance^kurto-signed_var^kurto-negative_signed_var^kurto-var"

dirs=${start}/GMMPolicy/${env}/^${start}/Savearray/Knack-exploration0.95/${env}/^${start}/SmallVariance0.95/${env}/^${start}/kurtosis-signed_variance0.95/${env}/^${start}/kurtosis-negative_signed_variance0.95/${env}/^${start}/kurtosis-small_variance0.95/${env}/
labels="Default^Knack^SmallVariance^kurto-signed_var^kurto-negative_signed_var^kurto-var"
mode="exploitation"

echo "--root-dirs ${dirs} --labels ${labels} --mode ${mode} --smooth 50"
python ../misc/plotter/return_plotter.py --root-dirs ${dirs} --labels ${labels} --mode ${mode} --smooth 50 --plot_mode iqr
