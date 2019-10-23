#!/usr/bin/env bash
PROJECTROOT=${HOME}/master/SubgoalGeneration
export PYTHONPATH=${PROJECTROOT}:$PYTHONPATH

#start=/mnt/ISINAS1/karino/SubgoalGeneration/data/MultipleKnack0.95

#start=/mnt/ISINAS1/karino/SubgoalGeneration/ParameterSearch/Savearray/Knack-exploration0.95
#start=/mnt/ISINAS1/karino/SubgoalGeneration/ExploitationRatioThreshold/Savearray/Knack-exploration/
start=/mnt/ISINAS1/karino/SubgoalGeneration/ExploitationRatioThreshold/Savearray/small_variance
#start=/mnt/ISINAS1/karino/SubgoalGeneration/ExploitationRatioThreshold/Savearray/GMMPolicy
#start=/mnt/ISINAS1/karino/SubgoalGeneration/ExploitationRatioThreshold/Savearray/EExploitation
#start=/mnt/ISINAS1/karino/SubgoalGeneration/ExploitationRatioThreshold/Savearray/signed_variance
#start=/mnt/ISINAS1/karino/SubgoalGeneration/ExploitationRatioThreshold/Savearray/negative_signed_variance
#env=HalfCheetah-v2
env=Walker2d-v2

dirs=`ls ${start}/${env}/ | grep seed`
for _dir in $dirs;
do
  echo ${start}/${env}/${_dir}
#  python ../../analysis/Q_function_analysis.py --env-id ${env} --policy-mode Knack-exploration --root-dir /tmp/test --entropy-coeff 0.0 --normalize-obs 0 --eval_n_episodes 20 --eval_n_frequency 1 --eval-model ${start}/${env}/${_dir}/model
  python ../../analysis/q_function_analysis_histogram.py --root_path ${start}/${env}/${_dir}
done
