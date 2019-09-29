#!/usr/bin/env bash
export PYTHONPATH=${HOME}/master/SubgoalGeneration:$PYTHONPATH

#env=Walker2d-v2
# knack max
#file=/mnt/ISINAS1/karino/SubgoalGeneration/data/MultipleKnack0.95/Walker2d-v2/seed3/model
# default_exploration max
#file=/mnt/ISINAS1/karino/SubgoalGeneration/data/improve_exploration/sac/GMMPolicy/Walker2d-v2/0719/seed5/model

env=HalfCheetah-v2
# knack max
#file=/mnt/ISINAS1/karino/SubgoalGeneration/data/MultipleKnack0.95/HalfCheetah-v2/seed3/model
# default_exploration max
file=/mnt/ISINAS1/karino/SubgoalGeneration/data/improve_exploration/sac/GMMPolicy/HalfCheetah-v2/0719/seed9/model

python ../experiments/make_expert_data.py --env-id ${env} --policy-mode Knack-exploration --normalize-obs 0 --entropy-coeff 0.0 --seed 0 --eval-model ${file}
