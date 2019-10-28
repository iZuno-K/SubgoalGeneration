#!/usr/bin/env bash
export PYTHONPATH=${HOME}/master/SubgoalGeneration:$PYTHONPATH

for seed in {1..100}
do
  python ../environments/maze.py --seed ${seed} --alg_type "Bottleneck" &
  python ../environments/maze.py --seed ${seed} --alg_type "EpsGreedy" &
  sleep 1
done
