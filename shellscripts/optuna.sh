#!/usr/bin/env bash
export PYTHONPATH=${HOME}/master/SubgoalGeneration:$PYTHONPATH
#env=Walker2d-v2
env=HalfCheetah-v2
e=0.3

# change sac/misc/tf_utils.py  also !!
thrednum=1
export MKL_NUM_THREADS=${thrednum}
export NUMEXPR_NUM_THREADS=${thrednum}
export OMP_NUM_THREADS=${thrednum}
#TODAY=`date "+%Y%m%d_%H%M"`

seed=1
eval_num=1
# change sac/misc/tf_utils.py  also !!
thrednum=1
export MKL_NUM_THREADS=${thrednum}
export NUMEXPR_NUM_THREADS=${thrednum}
export OMP_NUM_THREADS=${thrednum}
#policy_mode=Knack-exploration
#policy_mode=kurtosis-signed_variance
#policy_mode=kurtosis-negative_signed_variance
#policy_mode=kurtosis-small_variance
#policy_mode=kurtosis-negative_singed_variance_no_threshold
policy_mode=small_variance
savearray=0
LOGDIR1="/mnt/ISINAS1/karino/SubgoalGeneration/Optuna/${policy_mode}0.95/${env}/seed${seed}"
for thread in {1..5}
do
  CMD="python ../experiments/gym_experiment.py --env-id ${env} --policy-mode ${policy_mode} --root-dir ${LOGDIR1} --seed ${seed} --entropy-coeff 0.0 --n-epochs 2000 --normalize-obs 0 --eval_n_episodes ${eval_num} --eval_n_frequency 1 --save_array_flag ${savearray} --optuna"
  eval "${CMD}" &
  sleep 1
done
