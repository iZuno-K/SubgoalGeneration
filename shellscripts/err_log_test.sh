#!/usr/bin/env bash
export PYTHONPATH=${HOME}/master/SubgoalGeneration:$PYTHONPATH
#export PYTHONPATH=${HOME}/tmp_programs/SubgoalGeneration:$PYTHONPATH

env=Walker2d-v2
#env=HalfCheetah-v2
#env=MountainCa rContinuousOneTurn-v0
#env=Ant-v2


# change sac/misc/tf_utils.py  also !!
thrednum=2
export MKL_NUM_THREADS=${thrednum}
export NUMEXPR_NUM_THREADS=${thrednum}
export OMP_NUM_THREADS=${thrednum}
#TODAY=`date "+%Y%m%d_%H%M"`

eval_num=1
#policy_mode=Knack-exploration
#policy_mode=kurtosis-signed_variance
#policy_mode=kurtosis-negative_signed_variance
#policy_mode=kurtosis-small_variance
#policy_mode=kurtosis-negative_singed_variance_no_threshold
policy_mode=GMMPolicy
#policy_mode=EExploitation
e=0.95
#e=0.95  # if not EExploitation , always 0.95
savearray=0
ec=0.2

for seed in {1..10}
do
#  LOGDIR1="/mnt/ISINAS1/karino/SubgoalGeneration/ParameterSearch/Savearray/${policy_mode}${e}/${env}/seed${seed}"
#  LOGDIR1="/mnt/ISINAS1/karino/SubgoalGeneration/ParameterSearch/${policy_mode}/${env}/seed${seed}"  # GMMPolicy
  LOGDIR1="/mnt/ISINAS1/karino/SubgoalGeneration/ParameterSearch/EntBornus${ec}/${policy_mode}/${env}/seed${seed}"  # GMMPolicy
  CMD="python ../experiments/gym_experiment.py --env-id ${env} --policy-mode ${policy_mode} --root-dir ${LOGDIR1} --seed ${seed} --entropy-coeff ${ec} --n-epochs 2000 --normalize-obs 0 --eval_n_episodes ${eval_num} --eval_n_frequency 1 --save_array_flag ${savearray} --e ${e}"

#  LOGDIR1="/mnt/ISINAS1/karino/SubgoalGeneration/data/MultipleKnack0.95/ContinuousMazeTerminateDist/seed${seed}"
#  CMD="python ../experiments/continuous_maze.py --policy-mode Knack-exploration --root-dir ${LOGDIR1} --seed ${seed} --entropy-coeff 0.0 --n-epochs 2000 --path-mode DoubleRevised --reward-mode Dense --terminate-dist 0 --normalize-obs 0"

  eval "${CMD}" &
#  sleep 1
done
