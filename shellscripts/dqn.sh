#!/usr/bin/env bash
export PYTHONPATH=${HOME}/master/SubgoalGeneration:$PYTHONPATH

# change ../experiments/dqn_experiment.py num_cpu  also !!
thrednum=1
export MKL_NUM_THREADS=${thrednum}
export NUMEXPR_NUM_THREADS=${thrednum}
export OMP_NUM_THREADS=${thrednum}

#policy_mode=Default

policy_mode=large_variance
bottleneck_threshold_ratio=0.1


#feps=0.01
feps=0.0
frac=1.0

total_timesteps=2000000
#seed=1

env_id=BreakoutNoFrameskip-v4
#env_id=SeaquestNoFrameskip-v4

for seed in {15..25}
do
    # control method
#  LOGDIR1="/tmp/dqn_test2/seed${seed}"
#  LOGDIR1="/mnt/ISINAS1/karino/SubgoalGeneration/ExploitationRatioThreshold/DQNnew/plus1logging${total_timesteps}_feps${feps}_frac${frac}/${policy_mode}/${env_id}/seed${seed}"
#  python ../experiments/dqn_experiment.py --policy-mode ${policy_mode} --logdir ${LOGDIR1} --gpu 0 --seed ${seed}  --total_timesteps ${total_timesteps} --exploration_final_eps ${feps} --exploration_fraction ${frac} --use_my_env_wrapper --env_id ${env_id} &

  # proposed method
  LOGDIR1="/mnt/ISINAS1/karino/SubgoalGeneration/ExploitationRatioThreshold/DQNnew/plus1logging${total_timesteps}_feps${feps}_frac${frac}/${policy_mode}${bottleneck_threshold_ratio}/${env_id}/seed${seed}"
  python ../experiments/dqn_experiment.py --policy-mode ${policy_mode} --logdir ${LOGDIR1} --gpu 0 --seed ${seed} --exploitation_ratio_on_bottleneck 0.95 --bottleneck_threshold_ratio ${bottleneck_threshold_ratio} --total_timesteps ${total_timesteps} --exploration_final_eps ${feps} --exploration_fraction ${frac} --use_my_env_wrapper --env_id ${env_id} &
#  --save_array_flag
  sleep 1
done

#--root-dirs /mnt/ISINAS1/karino/SubgoalGeneration/ExploitationRatioThreshold/DQNnew/samelogging2000000_feps0.0_frac1.0/Default/BreakoutNoFrameskip-v4/^/mnt/ISINAS1/karino/SubgoalGeneration/ExploitationRatioThreshold/DQNnew/samelogging2000000_feps0.0_frac1.0/large_variance0.1/BreakoutNoFrameskip-v4/ --legends Default^large_variance --xlabel steps --ylabel eval return --smooth 50 --plot_mode iqr