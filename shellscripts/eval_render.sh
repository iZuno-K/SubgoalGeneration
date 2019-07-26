#!/usr/bin/env bash
export PYTHONPATH=${HOME}/master/SubgoalGeneration:$PYTHONPATH

ENV=Walker2d-v2
ENV=HalfCheetah-v2
python ../experiments/gym_experiment.py --env-id ${ENV} --policy-mode Knack-exploration --normalize-obs 0 --entropy-coeff 0.0 --seed 0 \
--eval-model /mnt/ISINAS1/karino/SubgoalGeneration/data/improve_exploration/sac/multiple_knack/${ENV}/0718/seed9/model
