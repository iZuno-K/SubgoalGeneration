#!/usr/bin/env bash
for i in {1..3}
do
python ../experiments/continuous_maze.py --root-dir /home/isi/karino/master/SubgoalGeneration/data/data7/sac --seed ${i} --entropy-coeff 0.0 --n-epochs 3000 --path-mode DoubleRevised &
sleep 1
done

# 毎エポックのstates knack の保存