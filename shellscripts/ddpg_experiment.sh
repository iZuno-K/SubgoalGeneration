#!/usr/bin/env bash
for i in {1..3}
do
python ../experiments/ddpg/main.py --env-id ContinuousSpaceMaze --path-mode Double --seed ${i} --nb-epochs 1000 --evaluation --save-dir ../data6/ddpg/ &
sleep 1
done
