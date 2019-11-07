#!/usr/bin/env bash

env=Walker2d-v2
start=/mnt/ISINAS1/karino/SubgoalGeneration/ExploitationRatioThreshold/Savearray
#start2=/mnt/ISINAS1/karino/SubgoalGeneration/ExploitationRatioThreshold/NoSavearray/

dirs=${start}/GMMPolicy/${env}/^${start}/EExploitation/${env}/^${start}/large_variance/${env}/
dirs=${start}/GMMPolicy/${env}/^${start}/EExploitation0.095/${env}/^${start}/large_variance0.1/${env}/
legends="Default^EExploitation^large_variance"

dirs=${start}/EExploitation0.095/${env}/^${start}/large_variance0.1/${env}/
legends="EExploitation^large_variance"

#dirs=${start2}/large_variance0.1/${env}/^${start2}/large_variance0.15/${env}/^${start}/large_variance/${env}/^${start2}/large_variance0.3/${env}/
#legends="0.1^0.15^0.2^0.3"


ylabel="eval_average_return"
xlabel="total_step"
echo "--root-dirs ${dirs} --legends ${legends} --xlabel ${xlabel} --ylabel ${ylabel} --smooth 50 --plot_mode iqr"
python ../../misc/plotter/return_plotter.py --root-dirs ${dirs} --legends ${legends} --xlabel ${xlabel} --ylabel ${ylabel} --smooth 10 --plot_mode iqr
#--save_path /home/karino/Desktop/ExploitationRatioThreshold/${env}.pdf

