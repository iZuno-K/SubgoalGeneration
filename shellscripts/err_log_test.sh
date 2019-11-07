#!/usr/bin/env bash
export PYTHONPATH=${HOME}/master/SubgoalGeneration:$PYTHONPATH
#export PYTHONPATH=${HOME}/tmp_programs/SubgoalGeneration:$PYTHONPATH

env=Walker2d-v2
#env=Walker2dWOFallReset-v0
#env=HalfCheetah-v2
#env=MountainCa rContinuousOneTurn-v0
#env=Hopper-v2
#env=Ant-v2
#env=Humanoid-v2

# change sac/misc/tf_utils.py  also !!
thrednum=4
export MKL_NUM_THREADS=${thrednum}
export NUMEXPR_NUM_THREADS=${thrednum}
export OMP_NUM_THREADS=${thrednum}

eval_num=1

policy_mode=GMMPolicy

#policy_mode=EExploitation
#exploitation_ratio=0.19  # 0.2 * 0.95
#exploitation_ratio=0.095  # 0.1 * 0.95

#policy_mode=large_variance
#exploitation_ratio=0.2
#exploitation_ratio=0.1

#e=0.95
reward_sckae=1.
#e=0.95  # if not EExploitation , always 0.95
savearray=1
ec=0.0

for seed in {21..25}
do
  LOGDIR1="/mnt/ISINAS1/karino/SubgoalGeneration/ExploitationRatioThreshold/Savearray/${policy_mode}/${env}/seed${seed}"
#  LOGDIR1="/mnt/ISINAS1/karino/SubgoalGeneration/ExploitationRatioThreshold/Savearray/${policy_mode}${exploitation_ratio}/${env}/seed${seed}"

  CMD="python ../experiments/gym_experiment.py --env-id ${env} --policy-mode ${policy_mode} --root-dir ${LOGDIR1} --seed ${seed} --entropy-coeff ${ec} --n-epochs 2000 --normalize-obs 0 --eval_n_episodes ${eval_num} --eval_n_frequency 1 --save_array_flag ${savearray} --exploitation_ratio ${exploitation_ratio}"

  eval "${CMD}" &
  sleep 1

done