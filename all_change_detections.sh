#!/bin/bash

# python run_change_metrics.py model="EDSR_UNet_SL1" nb_version=6
# python run_change_metrics.py model="RCAN_UNet_SL1" nb_version=4
# python run_change_metrics.py model="EDSR_UNet_SL1_res1-5" nb_version=1
# python run_change_metrics.py model="EDSR_UTAE_SL1_naive" nb_version=2
python run_change_metrics.py model="SuperResUnet_SL1" nb_version=4
# python run_change_metrics.py model="SuperResUnet_SL1_res1-5" nb_version=1
# python run_change_metrics.py model="UNet_SL1" nb_version=2
# python run_change_metrics.py model="UTAE_SL1" nb_version=3
# python run_change_metrics.py model="UTAE_SL1_16img" nb_version=5
# python run_change_metrics.py model="EDSR_UTAE_SL1_16img" nb_version=4
python run_change_metrics.py model="EDSR_UTAE_SL1_16img_res-inp-6" nb_version=1
# python run_change_metrics.py model="EDSR_UTAE_SL1_32img" nb_version=4


