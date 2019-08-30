#!/usr/bin/env bash
PROJECTROOT=${HOME}/master/SubgoalGeneration
export PYTHONPATH=${PROJECTROOT}:$PYTHONPATH

#env=HalfCheetah-v2
env=Walker2d-v2
#DATAROOT=/mnt/ISINAS1/karino/SubgoalGeneration/data/improve_exploration/sac/multiple_knack/${env}/
DATAROOT=/mnt/ISINAS1/karino/SubgoalGeneration/data/MultipleKnack0.95/${env}/
DEPTH=3
EPOCHID=1000
SAVEDIR=/mnt/ISINAS1/karino/SubgoalGeneration/data/MultipleKnack0.95/${env}
SAVEMODE=.pdf
#echo "python ${PROJECTROOT}/analysis/knack_plotter_episode.py --data_root_dir ${DATAROOT} --file_place_depth ${DEPTH} --target_epoch ${EPOCHID} --save_dir ${SAVEDIR} --save_mode ${SAVEMODE}"
python "${PROJECTROOT}/analysis/knack_plotter_episode.py" --data_root_dir ${DATAROOT} \
--file_place_depth ${DEPTH} --target_epoch ${EPOCHID} --save_dir ${SAVEDIR} --save_mode ${SAVEMODE}