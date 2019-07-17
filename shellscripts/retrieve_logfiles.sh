#!/usr/bin/env bash
# local
#export PYTHONPATH=/home/karino/Programs/master/SubgoalGeneration:$PYTHONPATH

# # remote server
export PYTHONPATH=/home/isi/karino/master/SubgoalGeneration:$PYTHONPATH
ROOTDIR=/tmp/karino/kanck/improve_exploration/sac/GMMPolicy
DEPTH=3
SAVEDIR=/tmp/karino/kanck/improve_exploration/sac/GMMPolicy_logonly
python /home/isi/karino/master/SubgoalGeneration/misc/retrieve_logfile.py --root-dir ${ROOTDIR} --depth ${DEPTH} --save-dir ${SAVEDIR}
#
# # dgx
#export PYTHONPATH=/home/karino/tmp_programs/SubgoalGeneration:$PYTHONPATH
#ROOTDIR=/home/karino/tmp_logfiles/improve_exploration/sac
#DEPTH=3
#SAVEDIR=/home/karino/tmp_logfiles/improve_exploration/sac_logonly
#python /home/karino/tmp_programs/SubgoalGeneration/misc/retrieve_logfile.py --root-dir ${ROOTDIR} --depth ${DEPTH} --save-dir ${SAVEDIR}
#
