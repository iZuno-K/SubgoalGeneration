#!/usr/bin/env bash
export PYTHONPATH=${HOME}/master/SubgoalGeneration:$PYTHONPATH
#export PYTHONPATH=${HOME}/tmp_programs/SubgoalGeneration:$PYTHONPATH

env=Walker2d-v2
#env=HalfCheetah-v2
#env=MountainCarContinuousOneTurn-v0
#env=Ant-v2
e=0.35

export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OMP_NUM_THREADS=2
#TODAY=`date "+%Y%m%d_%H%M"`
for seed in {1..7}
do
#  LOGDIR1="/tmp/data/MultipleKnack0.95/${env}/seed${seed}"
  LOGDIR1="/mnt/ISINAS1/karino/SubgoalGeneration/data/EExploitation/e${e}${env}/seed${seed}"
  CMD="python ../experiments/gym_experiment.py --env-id ${env} --policy-mode EExploitationPolicy --root-dir ${LOGDIR1} --seed ${seed} --entropy-coeff 0.0 --n-epochs 2000 --normalize-obs 0 --e ${e}"
#  LOGDIR1="/mnt/ISINAS1/karino/SubgoalGeneration/data/GMMPolicy/${env}/seed${seed}"
#  CMD="python ../experiments/gym_experiment.py --env-id ${env} --policy-mode GMMPolicy --root-dir ${LOGDIR1} --seed ${seed} --entropy-coeff 0.0 --n-epochs 2000 --normalize-obs 0 &"

#  LOGDIR1="/mnt/ISINAS1/karino/SubgoalGeneration/data/MultipleKnack0.95/${env}/seed${seed}"
#  CMD="python ../experiments/gym_experiment.py --env-id ${env} --policy-mode Knack-exploration --root-dir ${LOGDIR1} --seed ${seed} --entropy-coeff 0.0 --n-epochs 2000 --normalize-obs 0"

#  LOGDIR1="/mnt/ISINAS1/karino/SubgoalGeneration/data/MultipleKnack0.95/ContinuousMazeTerminateDist/seed${seed}"
#  CMD="python ../experiments/continuous_maze.py --policy-mode Knack-exploration --root-dir ${LOGDIR1} --seed ${seed} --entropy-coeff 0.0 --n-epochs 2000 --path-mode DoubleRevised --reward-mode Dense --terminate-dist 0 --normalize-obs 0"

#  mkdir -p ${LOGDIR1}
#  LOG_PATH="${LOGDIR1}/stdlog.log"
#  ERR_PATH="${LOGDIR1}/stderr.log"
#  echo "$CMD"
#  { { ${CMD} | tee -a ${LOG_PATH}; } 3>&2 2>&1 1>&3 | tee -a ${ERR_PATH}; } 3>&2 2>&1 1>&3 &
  sleep 10; eval "${CMD}" &
done