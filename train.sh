d2=`date +%Y_%m_%d_%H_%M_%S`
python train_da.py --s1 i --s2 p --t c --seed 83 --mode 1  --threshold 0.95 --on_lbd 0.0004 --push_scale 1.0 > logs/${d2}.txt 2>&1