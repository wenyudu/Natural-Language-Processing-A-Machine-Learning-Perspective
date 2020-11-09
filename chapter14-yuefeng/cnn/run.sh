#CNN with dropout set to 0.2 and filter_size/kernel_size set to 3
python train.py  --lr=0.001 --itr=20 --dropout=0.2 --device='cuda' --n_filters=5 --filter_size=3
