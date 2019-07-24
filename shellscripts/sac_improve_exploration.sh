#!/usr/bin/env bash
# local
#export PYTHONPATH=/home/karino/Programs/master/SubgoalGeneration:$PYTHONPATH
# remote server
export PYTHONPATH=/home/isi/karino/master/SubgoalGeneration:$PYTHONPATH
# dgx
#export PYTHONPATH=/home/karino/tmp_programs/SubgoalGeneration:$PYTHONPATH

#LOGDIR="/mnt/ISINAS1/karino/SubgoalGeneration/data/improve_exploration/sac/"

#LOGDIR1="/home/karino/tmp_logfiles/improve_exploration/sac/"
LOGDIR1="/tmp_karino/multiple_knack/improve_exploration/sac/"
#LOGDIR1="/tmp/multiple_knack/improve_exploration/sac/"
#i=4
#for i in {4..10}
#do
#python ../experiments/gym_experiment.py --env-id MountainCarContinuousOneTurn-v0 --policy-mode Knack-exploration --root-dir ${LOGDIR1} --seed ${i} --entropy-coeff 0.0 --n-epochs 2000 --normalize-obs 0 &
#sleep 1
#python ../experiments/gym_experiment.py --env-id HalfCheetah-v2 --policy-mode Knack-exploration --root-dir ${LOGDIR1} --seed ${i} --entropy-coeff 0.0 --n-epochs 2000 --normalize-obs 0 &
#sleep 1
#python ../experiments/gym_experiment.py --env-id Walker2d-v2 --policy-mode Knack-exploration --root-dir ${LOGDIR1} --seed ${i} --entropy-coeff 0.0 --n-epochs 2000 --normalize-obs 0 &
#sleep 1
#done

#LOGDIR2="/mnt/ISINAS1/karino/SubgoalGeneration/data/improve_exploration/sac/GMMPolicy"
LOGDIR2="/home/karino/tmp_logfiles/improve_exploration/sac/GMMPolicy"
#LOGDIR2="/tmp_karino/karino/kanck/improve_exploration/sac/GMMPolicy"

for i in {4..10}
do
#python ../experiments/gym_experiment.py --env-id MountainCarContinuousOneTurn-v0 --policy-mode GMMPolicy --root-dir ${LOGDIR2} --seed ${i} --entropy-coeff 0.0 --n-epochs 2000 --normalize-obs 0 &
#sleep 1
python ../experiments/gym_experiment.py --env-id HalfCheetah-v2 --policy-mode GMMPolicy --root-dir ${LOGDIR2} --seed ${i} --entropy-coeff 0.0 --n-epochs 2000 --normalize-obs 0 &
sleep 1
python ../experiments/gym_experiment.py --env-id Walker2d-v2 --policy-mode GMMPolicy --root-dir ${LOGDIR2} --seed ${i} --entropy-coeff 0.0 --n-epochs 2000 --normalize-obs 0 &
sleep 1
done


#for i in {1..2}
#do
#python ../experiments/gym_experiment.py --env-id Humanoid-v2 --policy-mode Knack-exploration --root-dir ${LOGDIR1} --seed ${i} --entropy-coeff 0.0 --n-epochs 2000 --normalize-obs 0 &
#sleep 1
#python ../experiments/gym_experiment.py --env-id Humanoid-v2 --policy-mode GMMPolicy --root-dir ${LOGDIR2} --seed ${i} --entropy-coeff 0.0 --n-epochs 2000 --normalize-obs 0 &
#sleep 1
#python ../experiments/gym_experiment.py --env-id AntNoForce-v0 --policy-mode Knack-exploration --root-dir ${LOGDIR1} --seed ${i} --entropy-coeff 0.0 --n-epochs 2000 --normalize-obs 0 &
#sleep 1
#python ../experiments/gym_experiment.py --env-id AntNoForce-v0 --policy-mode GMMPolicy --root-dir ${LOGDIR2} --seed ${i} --entropy-coeff 0.0 --n-epochs 2000 --normalize-obs 0 &
#sleep 1
#done
#
