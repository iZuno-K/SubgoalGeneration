#!/usr/bin/env bash

env=BreakoutNoFrameskip-v4
#start=/mnt/ISINAS1/karino/SubgoalGeneration/ExploitationRatioThreshold/DQN
#start2=/mnt/ISINAS1/karino/SubgoalGeneration/ExploitationRatioThreshold/DQN/samelogging/
#start2=/mnt/ISINAS1/karino/SubgoalGeneration/ExploitationRatioThreshold/DQN/samelogging2000000
start2=/mnt/ISINAS1/karino/SubgoalGeneration/ExploitationRatioThreshold/DQN/samelogging2000000_feps0.05

#dirs=${start}/Default/${env}/^${start}/large_variance/${env}/
#legends="Default^large_variance"

dirs=${start2}/Default/${env}/^${start2}/large_variance0.1/${env}/
legends="Default^large_variance"

#dirs=${start2}/Default/${env}/^${start2}/large_variance0.05/${env}/^${start2}/large_variance0.1/${env}/^${start2}/large_variance0.15/${env}/^${start2}/large_variance0.2/${env}/^${start2}/large_variance0.3/${env}/^${start2}/large_variance0.4/${env}/
#legends="Default^0.05^0.1^0.15^0.2^0.3^0.4"

#dirs=${start}/EExploitation/${env}/^${start}/large_variance/${env}/
#legends="EExploitation^large_variance"

ylabel="eval return"
#ylabel="mean 100 episode return"
xlabel="steps"

echo "--root-dirs ${dirs} --legends ${legends} --xlabel ${xlabel} --ylabel ${ylabel} --smooth 50 --plot_mode raw"
#python ../../misc/plotter/return_plotter.py --root-dirs ${dirs} --legends ${legends} --xlabel ${xlabel} --ylabel ${ylabel} --smooth 100 --plot_mode raw
#--save_path /home/karino/Desktop/ExploitationRatioThreshold/${env}.pdf



#--root-dirs /mnt/ISINAS1/karino/SubgoalGeneration/ExploitationRatioThreshold/DQN/samelogging/Default/BreakoutNoFrameskip-v4/samelogging2000000^/mnt/ISINAS1/karino/SubgoalGeneration/ExploitationRatioThreshold/DQN/samelogging2000000//large_variance0.1/BreakoutNoFrameskip-v4/ --legends Default^large_variance --xlabel steps --ylabel eval return --smooth 50 --plot_mode iqr