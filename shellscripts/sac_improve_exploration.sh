#!/usr/bin/env bash
#export PYTHONPATH=/home/karino/Programs/master/SubgoalGeneration:$PYTHONPATH
export PYTHONPATH=/home/isi/karino/master/SubgoalGeneration:$PYTHONPATH

LOGDIR="/mnt/ISINAS1/karino/SubgoalGeneration/data/improve_exploration/sac/"

for i in {1..3}
do
#python ../experiments/gym_experiment.py --env-id MountainCarContinuous-v0 --root-dir ${logdir} --seed ${i} --entropy-coeff 0.0 --n-epochs 2000 --normalize-obs 0 &
#sleep 1
python ../experiments/gym_experiment.py --env-id MountainCarContinuousOneTurn-v0 --policy-mode Knack-exploration --root-dir ${LOGDIR} --seed ${i} --entropy-coeff 0.0 --n-epochs 2000 --normalize-obs 0 &
sleep 1
python ../experiments/gym_experiment.py --env-id HalfCheetah-v2 --policy-mode Knack-exploration --root-dir ${LOGDIR} --seed ${i} --entropy-coeff 0.0 --n-epochs 2000 --normalize-obs 0 &
sleep 1
python ../experiments/gym_experiment.py --env-id Walker-v2 --policy-mode Knack-exploration --root-dir ${LOGDIR} --seed ${i} --entropy-coeff 0.0 --n-epochs 2000 --normalize-obs 0 &
sleep 1
#python ../experiments/continuous_maze.py --root-dir ${logdir} --seed ${i} --entropy-coeff 0.0 --n-epochs 2000 --path-mode DoubleRevised --reward-mode Dense --terminate-dist 0 --normalize-obs 0 --opt-log-name KnackPControl --policy-mode Knack-p_control &
#sleep 1
#python ../experiments/continuous_maze.py --root-dir ${logdir} --seed ${i} --entropy-coeff 0.0 --n-epochs 2000 --path-mode DoubleRevised --reward-mode Dense --terminate-dist 0 --normalize-obs 0 --opt-log-name KnackExploitation --policy-mode Knack-exploitation &
#sleep 1
#python ../experiments/continuous_maze.py --root-dir ${logdir} --seed ${i} --entropy-coeff 0.0 --n-epochs 2000 --path-mode DoubleRevised --reward-mode Dense --terminate-dist 0 --normalize-obs 0 --opt-log-name KnackExploration --policy-mode Knack-exploration &
#sleep 1
done


