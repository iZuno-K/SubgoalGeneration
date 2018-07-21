#!/usr/bin/env bash
python ../misc/plotter.py --root-dir ../data5/DoubleContinuousSpaceMaze20_45_RB1e6_entropy0.0_epoch3000__Normalize_dynamicCoeff/seed1 &
sleep 30s
python ../misc/plotter.py --root-dir ../data5/DoubleContinuousSpaceMaze20_45_RB1e6_entropy1.0_epoch3000__Normalize_dynamicCoeff/seed1 &
sleep 30s
python ../misc/plotter.py --root-dir ../data5/EasierDoubleContinuousSpaceMaze20_45_RB1e6_entropy0.0_epoch3000__Normalize_dynamicCoeff/seed1 &
sleep 30s
python ../misc/plotter.py --root-dir ../data5/EasierDoubleContinuousSpaceMaze20_45_RB1e6_entropy1.0_epoch3000__Normalize_dynamicCoeff/seed1 &
sleep 30s
python ../misc/plotter.py --root-dir ../data5/SinglePathContinuousSpaceMaze20_45_RB1e6_entropy0.0_epoch3000__Normalize_dynamicCoeff/seed1 &
sleep 30s
python ../misc/plotter.py --root-dir ../data5/SinglePathContinuousSpaceMaze20_45_RB1e6_entropy1.0_epoch3000__Normalize_dynamicCoeff/seed1 &