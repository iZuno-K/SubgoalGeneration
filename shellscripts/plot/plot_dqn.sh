#!/usr/bin/env bash

env=BreakoutNoFrameskip-v4
start=/mnt/ISINAS1/karino/SubgoalGeneration/ExploitationRatioThreshold/DQN

dirs=${start}/Default/${env}/^${start}/large_variance/${env}/
legends="Default^large_variance"

#dirs=${start}/EExploitation/${env}/^${start}/large_variance/${env}/
#legends="EExploitation^large_variance"

ylabel="eval return"
ylabel="mean 100 episode return"
xlabel="steps"

python ../../misc/plotter/return_plotter.py --root-dirs ${dirs} --legends ${legends} --xlabel ${xlabel} --ylabel ${ylabel} --smooth 50 --plot_mode raw
#--save_path /home/karino/Desktop/ExploitationRatioThreshold/${env}.pdf