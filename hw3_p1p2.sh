

wget https://www.dropbox.com/s/r23z419e3kh0uai/acgan_bestmodel.pth?dl=0 -O './acgan_generator_model'

wget https://www.dropbox.com/s/8c9d6iy6avt4dmd/gan_bestmodel.pth?dl=0 -O './gan_generator_model'



python3 smile_gan_acgan.py --resume1 './gan_generator_model' --resume2 './acgan_generator_model' --save_dir $1