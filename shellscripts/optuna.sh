#!/usr/bin/env bash
export PYTHONPATH=${HOME}/master/SubgoalGeneration:$PYTHONPATH
#env=Walker2d-v2
env=HalfCheetah-v2
e=0.3

seed=1
eval_num=1
# change sac/misc/tf_utils.py  also !!
thrednum=4
export MKL_NUM_THREADS=${thrednum}
export NUMEXPR_NUM_THREADS=${thrednum}
export OMP_NUM_THREADS=${thrednum}
#policy_mode=Knack-exploration
#policy_mode=negative_signed_variance
#policy_mode=small_variance
policy_mode=signed_variance

savearray=0
exploitation_ratio=0.2
#LOGDIR1="/mnt/ISINAS1/karino/SubgoalGeneration/Optuna/test/${policy_mode}0.95/${env}/seed${seed}"
LOGDIR1="/tmp/Optuna/${policy_mode}/${env}/seed${seed}"
for thread in {1..3}
do
  CMD="python ../experiments/gym_experiment.py --env-id ${env} --policy-mode ${policy_mode} --root-dir ${LOGDIR1} --seed ${seed} --entropy-coeff 0.0 --n-epochs 2000 --normalize-obs 0 --eval_n_episodes ${eval_num} --eval_n_frequency 1 --save_array_flag ${savearray} --use_optuna --exploitation_ratio ${exploitation_ratio}"
  eval "${CMD}" &
  sleep 1
done
