#!/usr/bin/env bash
for i in {1..3}
do
python ../experiments/continuous_maze.py --root-dir ../data6 --seed ${i} --entropy-coeff 1.0 --n-epochs 3000 --dynamic-coeff True --path-mode Double &
sleep 1
done