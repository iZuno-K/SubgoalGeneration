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
# for i in {1..3}
# do
# python ../experiments/ddpg/main.py --env-id ContinuousSpaceMaze --path-mode DoubleRevised --seed ${i} --nb-epochs 3000 --evaluation --save-dir /mnt/qnap_o1/karino/knack_experiments/ddpg &
# sleep 1
# done

# 2018/09/18
# action noiseで
# for i in {1..3}
# do
# python ../experiments/ddpg/main.py --env-id ContinuousSpaceMaze --path-mode DoubleRevised --seed ${i} --nb-epochs 3000 --evaluation \
# --save-dir /mnt/qnap_o1/karino/knack_experiments/ddpg --noise-type normal_0.2 --opt-log-name action_noise &
# sleep 1
# done

# 2018/09/19
# clip norm で安定させられる？
for i in {1..3}
do
python ../experiments/ddpg/main.py --env-id ContinuousSpaceMaze --path-mode DoubleRevised --seed ${i} --nb-epochs 3000 --evaluation \
--save-dir /mnt/qnap_o1/karino/knack_experiments/ddpg --noise-type normal_0.2 --clip-norm 1.0 --opt-log-name clip_norm &
sleep 1
done

# これより前はmemory.saveに関するknack計算をnormalizeしないでやってた