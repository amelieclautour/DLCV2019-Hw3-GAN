wget https://www.dropbox.com/s/ro6m0tqvqu7ifcd/domainadaptation2_model28-0.pth?dl=0 -O './domain_adaptation_model'

python3 preds_label.py --data_test_dir $1 --target $2  --save_dir $3 --resume './domain_adaptation_model'

