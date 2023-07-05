# DREAM-Net for CT
An modified version of FISTA-Net based on https://github.com/jinxixiang/FISTA-Net

1. Install torch-randon (https://github.com/matteo-ronchetti/torch-radon)
2. Generate simulated sparse-view CT data: gen_ld_data.py
3. Generate training data: make_proj_img_list.py
4. Train and Validation: train_firstnet.py
5. Test: test_firstnet.py
