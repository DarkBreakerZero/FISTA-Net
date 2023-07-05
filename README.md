# FISTA-Net for CT
An modified version of FISTA-Net based on https://github.com/jinxixiang/FISTA-Net

1. Install torch-randon (https://github.com/matteo-ronchetti/torch-radon)
2. Generate simulated sparse-view CT data: gen_ld_data.py
3. Generate training data: make_proj_img_list.py
4. Train and Validation: train_firstnet.py
5. Test: test_firstnet.py

Please cite the following references:
1. TorchRadon: Fast Differentiable Routines for Computed Tomography
2. A model-based deep learning network for inverse problem in imaging
3. DREAM-Net: Deep Residual Error iterAtive Minimization Network for Sparse-View CT Reconstruction
