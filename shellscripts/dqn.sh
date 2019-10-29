#!/usr/bin/env bash
export PYTHONPATH=${HOME}/master/SubgoalGeneration:$PYTHONPATH

# change ../experiments/dqn_experiment.py num_cpu  also !!
thrednum=1
export MKL_NUM_THREADS=${thrednum}
export NUMEXPR_NUM_THREADS=${thrednum}
export OMP_NUM_THREADS=${thrednum}

policy_mode=large_variance
#policy_mode=Default

#seed=1
for seed in {1..6}
do
  LOGDIR1="/mnt/ISINAS1/karino/SubgoalGeneration/ExploitationRatioThreshold/DQN/${policy_mode}/BreakoutNoFrameskip-v4/seed${seed}"
#  LOGDIR1="/tmp/dqn"
#  python ../experiments/dqn_experiment.py --policy-mode ${policy_mode} --logdir ${LOGDIR1} --seed ${seed} &
  python ../experiments/dqn_experiment.py --policy-mode ${policy_mode} --logdir ${LOGDIR1} --seed ${seed} --exploitation_ratio_on_bottleneck 0.95 --bottleneck_threshold_ratio 0.2 &
  sleep 1
done
