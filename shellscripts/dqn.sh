#!/usr/bin/env bash
export PYTHONPATH=${HOME}/master/SubgoalGeneration:$PYTHONPATH

# change ../experiments/dqn_experiment.py num_cpu  also !!
thrednum=1
export MKL_NUM_THREADS=${thrednum}
export NUMEXPR_NUM_THREADS=${thrednum}
export OMP_NUM_THREADS=${thrednum}

policy_mode=large_variance
policy_mode=Default
#e=0.95  # if not EExploitation , always 0.95

#seed=1
for seed in {1..6}
do
  LOGDIR1="/mnt/ISINAS1/karino/SubgoalGeneration/ExploitationRatioThreshold/DQN/${policy_mode}/BreakoutNoFrameskip-v4/seed${seed}"
#  LOGDIR1="/tmp/dqn"
  python ../experiments/dqn_experiment.py --policy-mode ${policy_mode} --logdir ${LOGDIR1} --seed ${seed} &
  sleep 1
done
