#!/usr/bin/env bash

env=Walker2d-v2
start=/mnt/ISINAS1/karino/SubgoalGeneration/ExploitationRatioThreshold/Savearray

dirs=${start}/GMMPolicy/${env}/^${start}/EExploitation/${env}/^${start}/large_variance/${env}/
legends="Default^EExploitation^large_variance"



ylabel="eval_average_return"
xlabel="total_step"
echo "--root-dirs ${dirs} --legends ${legends} --xlabel ${xlabel} --ylabel ${ylabel} --smooth 50 --plot_mode iqr"
python ../../misc/plotter/return_plotter.py --root-dirs ${dirs} --legends ${legends} --xlabel ${xlabel} --ylabel ${ylabel} --smooth 50 --plot_mode iqr
#--save_path /home/karino/Desktop/ExploitationRatioThreshold/${env}.pdf