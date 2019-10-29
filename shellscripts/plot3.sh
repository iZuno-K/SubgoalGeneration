#!/usr/bin/env bash

#env=MountainCarContinuousOneTurn-v0
env=HalfCheetah-v2
#env=Walker2d-v2
#env=Walker2dWOFallReset-v0
#env=Hopper-v2
#env=Ant-v2

#START=/mnt/ISINAS1/karino/SubgoalGeneration/data
#start=/mnt/ISINAS1/karino/SubgoalGeneration/ParameterSearch
start=/mnt/ISINAS1/karino/SubgoalGeneration/ExploitationRatioThreshold/Savearray
#dirs=${START}/improve_exploration/sac/GMMPolicy/${env}/0719/^${START}/EExploitation/e0.3${env}/^${START}/EExploitation/e0.35${env}/^${START}/EExploitation/e0.4${env}/^${START}/MultipleKnack0.95/${env}/^${START}/SingedVariance0.95/${env}/^${START}/NegativeSingedVariance0.95/${env}/^${start}/SmallVariance0.95/${env}/^${start}/kurtosis-signed_variance0.95/${env}/^${start}/kurtosis-negative_signed_variance0.95/${env}/^${start}/kurtosis-small_variance0.95/${env}/
#labels="DefaultExploration^EExploitation0.3^EExploitation0.35^EExploitation0.4^KnackExploration^SignedVariance^NegativeSignedVariance^SmallVariance^kurto-signed_var^kurto-negative_signed_var^kurto-var"

# analyze kurtosis
#dirs=${start}/GMMPolicy/${env}/^${start}/Savearray/Knack-exploration0.95/${env}/^${start}/SmallVariance0.95/${env}/^${START}/NegativeSingedVariance0.95/${env}/^${start}/kurtosis-signed_variance0.95/${env}/^${start}/kurtosis-negative_signed_variance0.95/${env}/^${start}/kurtosis-small_variance0.95/${env}/
#labels="Default^Knack^SmallVariance^NegativeSignedVar^kurto-signed_var^kurto-negative_signed_var^kurto-smallvar"
#mode="exploitation"

# with entropy bornus
dirs=${start}/GMMPolicy/${env}/^${start}/Knack-exploration/${env}/^${start}/small_variance/${env}/^${start}/signed_variance/${env}/^${start}/large_variance/${env}/^${start}/EExploitation/${env}/
dirs=${start}/GMMPolicy/${env}/^${start}/EExploitation/${env}/^${start}/large_variance/${env}/

#^${start}/negative_signed_variance/${env}/
#labels="Default^kurtosis^small_variance^signed_variance^large_variance^EExploitation"
labels="Default^EExploitation^large_variance"
#^negative_signed_variance"
mode="exploitation"
#mode="total_episode"

echo "--root-dirs ${dirs} --labels ${labels} --mode ${mode} --smooth 2"
python ../misc/plotter/return_plotter.py --root-dirs ${dirs} --labels ${labels} --mode ${mode} --smooth 50 --plot_mode raw
#--save_path /home/karino/Desktop/ExploitationRatioThreshold/${env}.pdf
