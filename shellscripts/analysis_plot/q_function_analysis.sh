#!/usr/bin/env bash
PROJECTROOT=${HOME}/master/SubgoalGeneration
export PYTHONPATH=${PROJECTROOT}:$PYTHONPATH

start=/mnt/ISINAS1/karino/SubgoalGeneration/data/MultipleKnack0.95
#env=HalfCheetah-v2
env=Walker2d-v2

dirs=`ls ${start}/${env}/ | grep seed`
for _dir in $dirs;
do
  echo ${start}/${env}/${_dir}
  python ../../analysis/Q_function_analysis.py --env-id ${env} --policy-mode Knack-exploration --root-dir /tmp/test --entropy-coeff 0.0 --normalize-obs 0 --eval_n_episodes 20 --eval_n_frequency 1 --eval-model ${start}/${env}/${_dir}/model
done
