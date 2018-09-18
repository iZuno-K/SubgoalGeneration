#!/usr/bin/env bash

#for i in {1..3}
#do
#python ../experiments/ddpg/main.py --env-id ContinuousSpaceMaze --path-mode Double --seed ${i} --nb-epochs 3000 --evaluation --save-dir ../data/data7/ddpg/ &
#sleep 1
#done

# 2018/09/16
# 迷路の左側すり抜けられないようにして実験
#for i in {1..3}
#do
#python ../experiments/ddpg/main.py --env-id ContinuousSpaceMaze --path-mode DoubleRevised --seed ${i} --nb-epochs 3000 --evaluation --save-dir ../data/data7/ddpg/0916 &
#sleep 1
#done

# 2018/09/18
# 経験した状態とそのコツドをすべて保存する
for i in {1..3}
do
python ../experiments/ddpg/main.py --env-id ContinuousSpaceMaze --path-mode DoubleRevised --seed ${i} --nb-epochs 3000 --evaluation --save-dir /mnt/qnap_o1/karino/knack_experiments/ddpg &
sleep 1
done