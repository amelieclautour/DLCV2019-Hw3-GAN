wget https://www.dropbox.com/s/9jf0wqafdy16icr/uda_domainadaptation_model_MNISTtoSVHN4_.pth?dl=0 -O './uda_domain_adaptation_model'



python3 preds_label.py --data_test_dir $1 --target $2  --save_dir $3 --resume './uda_domain_adaptation_model'



