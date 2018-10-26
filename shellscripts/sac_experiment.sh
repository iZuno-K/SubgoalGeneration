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


#logdir="/home/isi/karino/master/SubgoalGeneration/data/data7/sac/"
#for i in {1..3}
#do
# 2018/10/20 clip norm sac
#python ../experiments/continuous_maze.py --root-dir ${logdir} --seed ${i} --entropy-coeff 0.0 --n-epochs 3000 --path-mode DoubleRevised --reward-mode Dense --terminate-dist 0 --clip-norm 1.0 --opt-log-name clip_norm &
# 2018/10/22 save positive state --> bug: cannot detect reached goal
#python ../experiments/continuous_maze.py --root-dir ${logdir} --seed ${i} --entropy-coeff 0.0 --n-epochs 3000 --path-mode DoubleRevised --reward-mode Dense --terminate-dist 1 &
#sleep 1
#python ../experiments/continuous_maze.py --root-dir ${logdir} --seed ${i} --entropy-coeff 0.0 --n-epochs 3000 --path-mode DoubleRevised --reward-mode Dense --terminate-dist 1 --normalize-obs 0 --opt-log-name no_normalize &
#sleep 1
# 2018/10/23 --> bug not fixed
#python ../experiments/continuous_maze.py --root-dir ${logdir} --seed ${i} --entropy-coeff 0.0 --n-epochs 3000 --path-mode DoubleRevised --reward-mode Dense --terminate-dist 1 --opt-log-name bugfix &
#sleep 1
#python ../experiments/continuous_maze.py --root-dir ${logdir} --seed ${i} --entropy-coeff 0.0 --n-epochs 3000 --path-mode DoubleRevised --reward-mode Dense --terminate-dist 1 --normalize-obs 0 --opt-log-name bugfix_no_normalize &
#sleep 1
# 2018/10/24
#python ../experiments/continuous_maze.py --root-dir ${logdir} --seed ${i} --entropy-coeff 0.0 --n-epochs 2000 --path-mode DoubleRevised --reward-mode Dense --terminate-dist 1 --opt-log-name bugfixdone &
#sleep 1
#python ../experiments/continuous_maze.py --root-dir ${logdir} --seed ${i} --entropy-coeff 0.0 --n-epochs 2000 --path-mode DoubleRevised --reward-mode Dense --terminate-dist 1 --normalize-obs 0 --opt-log-name bugfixdone_no_normalize &
#sleep 1
#python ../experiments/continuous_maze.py --root-dir ${logdir} --seed ${i} --entropy-coeff 0.0 --n-epochs 2000 --path-mode DoubleRevised --reward-mode Dense --terminate-dist 0 --opt-log-name bugfixdone &
#sleep 1
#python ../experiments/continuous_maze.py --root-dir ${logdir} --seed ${i} --entropy-coeff 0.0 --n-epochs 2000 --path-mode DoubleRevised --reward-mode Dense --terminate-dist 0 --normalize-obs 0 --opt-log-name bugfixdone_no_normalize &
#sleep 1
#python ../experiments/continuous_maze.py --root-dir ${logdir} --seed ${i} --entropy-coeff 0.0 --n-epochs 2000 --path-mode DoubleRevised --reward-mode Dense --terminate-dist 1 --normalize-obs 0 --opt-log-name completeBugfixno_normalize &
#sleep 1
#python ../experiments/continuous_maze.py --root-dir ${logdir} --seed ${i} --entropy-coeff 0.0 --n-epochs 2000 --path-mode DoubleRevised --reward-mode Dense --terminate-dist 0 --normalize-obs 0 --opt-log-name completeBugfixno_normalize &
#sleep 1
#done

# 2018/10/25
logdir="/home/isi/karino/master/SubgoalGeneration/data/data7/sac/"

for i in {1..3}
do
#python ../experiments/gym_experiment.py --env-id MountainCarContinuous-v0 --root-dir ${logdir} --seed ${i} --entropy-coeff 0.0 --n-epochs 2000 --normalize-obs 0 &
#sleep 1
python ../experiments/gym_experiment.py --env-id MountainCarContinuousOneTurn-v0 --root-dir ${logdir} --seed ${i} --entropy-coeff 0.0 --n-epochs 2000 --normalize-obs 0 &
sleep 1
#python ../experiments/gym_experiment.py --env-id HalfCheetah-v2 --root-dir ${logdir} --seed ${i} --entropy-coeff 0.0 --n-epochs 2000 --normalize-obs 0 &
#sleep 1
#python ../experiments/continuous_maze.py --root-dir ${logdir} --seed ${i} --entropy-coeff 0.0 --n-epochs 2000 --path-mode DoubleRevised --reward-mode Dense --terminate-dist 0 --normalize-obs 0 --opt-log-name KnackPControl --policy-mode Knack-p_control &
#sleep 1
#python ../experiments/continuous_maze.py --root-dir ${logdir} --seed ${i} --entropy-coeff 0.0 --n-epochs 2000 --path-mode DoubleRevised --reward-mode Dense --terminate-dist 0 --normalize-obs 0 --opt-log-name KnackExploitation --policy-mode Knack-exploitation &
#sleep 1
#python ../experiments/continuous_maze.py --root-dir ${logdir} --seed ${i} --entropy-coeff 0.0 --n-epochs 2000 --path-mode DoubleRevised --reward-mode Dense --terminate-dist 0 --normalize-obs 0 --opt-log-name KnackExploration --policy-mode Knack-exploration &
#sleep 1
done


