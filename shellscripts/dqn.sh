#!/usr/bin/env bash
export PYTHONPATH=${HOME}/master/SubgoalGeneration:$PYTHONPATH

# change ../experiments/dqn_experiment.py num_cpu  also !!
thrednum=1
export MKL_NUM_THREADS=${thrednum}
export NUMEXPR_NUM_THREADS=${thrednum}
export OMP_NUM_THREADS=${thrednum}

#policy_mode=Default
policy_mode=large_variance

bottleneck_threshold_ratio=0.3

#seed=1
for seed in {1..6}
do
#  LOGDIR1="/tmp/dqn"
#  LOGDIR1="/mnt/ISINAS1/karino/SubgoalGeneration/ExploitationRatioThreshold/DQN/samelogging/${policy_mode}/BreakoutNoFrameskip-v4/seed${seed}"
#  python ../experiments/dqn_experiment.py --policy-mode ${policy_mode} --logdir ${LOGDIR1} --gpu 7 --seed ${seed} &

  LOGDIR1="/mnt/ISINAS1/karino/SubgoalGeneration/ExploitationRatioThreshold/DQN/samelogging/${policy_mode}${bottleneck_threshold_ratio}/BreakoutNoFrameskip-v4/seed${seed}"
  python ../experiments/dqn_experiment.py --policy-mode ${policy_mode} --logdir ${LOGDIR1} --gpu 0 --seed ${seed} --exploitation_ratio_on_bottleneck 0.95 --bottleneck_threshold_ratio ${bottleneck_threshold_ratio} &
#  --save_array_flag
  sleep 1
done
