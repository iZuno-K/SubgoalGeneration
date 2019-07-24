#!/usr/bin/env bash
export PYTHONPATH=/home/karino/Programs/master/SubgoalGeneration:$PYTHONPATH

ENV=Walker2d-v2
python ../experiments/gym_experiment.py --env-id ${ENV} --policy-mode Knack-exploration --normalize-obs 0 --entropy-coeff 0.0 --seed 0 \
--eval-model /mnt/ISINAS1/karino/SubgoalGeneration/multiple_knack/improve_exploration/sac/${ENV}/0718/seed9/model