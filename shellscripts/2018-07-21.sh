#!/usr/bin/env bash
for i in {1..3}
#for i in 1
do
python ../experiments/exp_continuous_mountain_car.py --root-dir ../data6 --seed ${i} --entropy-coeff 0.0 --n-epochs 1000 --dynamic-coeff False &
sleep 1
python ../experiments/exp_continuous_mountain_car.py --root-dir ../data6 --seed ${i} --entropy-coeff 1.0 --n-epochs 1000 --dynamic-coeff True  &
sleep 1

done