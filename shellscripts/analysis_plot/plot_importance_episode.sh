#!/usr/bin/env bash
PROJECTROOT=${HOME}/master/SubgoalGeneration
export PYTHONPATH=${PROJECTROOT}:$PYTHONPATH

DATAROOT=/mnt/ISINAS1/karino/SubgoalGeneration/data/improve_exploration/sac/multiple_knack/HalfCheetah-v2/
DEPTH=3
EPOCHID=1000
SAVEDIR=/mnt/ISINAS1/karino/SubgoalGeneration/multiple_knack/HalfCheeta-v2
SAVEMODE=.pdf
#echo "python ${PROJECTROOT}/analysis/knack_plotter_episode.py --data_root_dir ${DATAROOT} --file_place_depth ${DEPTH} --target_epoch ${EPOCHID} --save_dir ${SAVEDIR} --save_mode ${SAVEMODE}"
python "${PROJECTROOT}/analysis/knack_plotter_episode.py" --data_root_dir ${DATAROOT} \
--file_place_depth ${DEPTH} --target_epoch ${EPOCHID} --save_dir ${SAVEDIR} --save_mode ${SAVEMODE}