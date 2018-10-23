#!/usr/bin/env bash
#for i in {1..3}
#do
#python ../experiments/continuous_maze.py --root-dir /home/isi/karino/master/SubgoalGeneration/data/data7/sac --seed ${i} --entropy-coeff 0.0 --n-epochs 3000 --path-mode DoubleRevised &
#sleep 1
#done

# 毎エポックのstates knack の保存

#for i in {1..3}
#do
#python ../experiments/continuous_maze.py --root-dir /home/isi/karino/master/SubgoalGeneration/data/data7/sac/clip_norm --seed ${i} --entropy-coeff 0.0 --n-epochs 3000 --path-mode DoubleRevised --clip-norm 1.0 &
#sleep 1
#done

# 2018/10/19
#logdir="/home/isi/karino/master/SubgoalGeneration/data/data7/sac/"
#for i in {1..3}
#do
#python ../experiments/continuous_maze.py --root-dir ${logdir} --seed ${i} --entropy-coeff 0.0 --n-epochs 3000 --path-mode DoubleRevised --reward-mode Dense --terminate-dist 0 &
#sleep 1
#python ../experiments/continuous_maze.py --root-dir ${logdir} --seed ${i} --entropy-coeff 0.0 --n-epochs 3000 --path-mode DoubleRevised --reward-mode Sparse --terminate-dist 0 &
#sleep 1
#python ../experiments/continuous_maze.py --root-dir ${logdir} --seed ${i} --entropy-coeff 0.0 --n-epochs 3000 --path-mode DoubleRevised --reward-mode Dense --terminate-dist 1 &
#sleep 1
#python ../experiments/continuous_maze.py --root-dir ${logdir} --seed ${i} --entropy-coeff 0.0 --n-epochs 3000 --path-mode DoubleRevised --reward-mode Sparse --terminate-dist 1 &
#sleep 1
#done


logdir="/home/isi/karino/master/SubgoalGeneration/data/data7/sac/"
for i in {1..3}
do
# 2018/10/20 clip norm sac
#python ../experiments/continuous_maze.py --root-dir ${logdir} --seed ${i} --entropy-coeff 0.0 --n-epochs 3000 --path-mode DoubleRevised --reward-mode Dense --terminate-dist 0 --clip-norm 1.0 --opt-log-name clip_norm &
# 2018/10/22 save positive state
python ../experiments/continuous_maze.py --root-dir ${logdir} --seed ${i} --entropy-coeff 0.0 --n-epochs 3000 --path-mode DoubleRevised --reward-mode Dense --terminate-dist 1 &
sleep 1
python ../experiments/continuous_maze.py --root-dir ${logdir} --seed ${i} --entropy-coeff 0.0 --n-epochs 3000 --path-mode DoubleRevised --reward-mode Dense --terminate-dist 1 --normalize-obs 0 --opt-log-name no_normalize &
sleep 1
done
