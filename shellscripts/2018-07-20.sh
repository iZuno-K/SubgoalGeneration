#!/usr/bin/env bash
#for i in {1..3}
for i in 1
do
#python ../experiments/continuous_maze.py --root-dir ../data5 --seed ${i} --entropy-coeff 0.0 --n-epochs 3000 --dynamic-coeff False --single-path True  &
#sleep 1
#python ../experiments/continuous_maze.py --root-dir ../data5 --seed ${i} --entropy-coeff 1.0 --n-epochs 3000 --dynamic-coeff True --single-path True &
#sleep 1

python ../experiments/continuous_maze.py --root-dir ../data5 --seed ${i} --entropy-coeff 0.0 --n-epochs 3000 --dynamic-coeff False --path-mode Double  &
sleep 1
python ../experiments/continuous_maze.py --root-dir ../data5 --seed ${i} --entropy-coeff 1.0 --n-epochs 3000 --dynamic-coeff True --path-mode Double &
sleep 1
python ../experiments/continuous_maze.py --root-dir ../data5 --seed ${i} --entropy-coeff 0.0 --n-epochs 3000 --dynamic-coeff False --path-mode OneHole  &
sleep 1
python ../experiments/continuous_maze.py --root-dir ../data5 --seed ${i} --entropy-coeff 1.0 --n-epochs 3000 --dynamic-coeff True --path-mode OneHole &
sleep 1
python ../experiments/continuous_maze.py --root-dir ../data5 --seed ${i} --entropy-coeff 0.0 --n-epochs 3000 --dynamic-coeff False --path-mode EasierDouble  &
sleep 1
python ../experiments/continuous_maze.py --root-dir ../data5 --seed ${i} --entropy-coeff 1.0 --n-epochs 3000 --dynamic-coeff True --path-mode EasierDouble &
sleep 1

done