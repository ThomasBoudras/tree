#!/bin/sh
### Note: No commands may be executed until after the #PBS lines
### Account information
#PBS -W group_list=ku_00196 -A ku_00196
### Job name (comment out the next line to get the name of the script used as the job name)
#PBS -N RCAN_UNet_SL1
### Output files (comment out the next 2 lines to get the job name used instead)
#PBS -j oe
#PBS -o logs_qsub/RCAN_UNet_SL1.log
### Only send mail when job is aborted or terminates abnormally
#PBS -m e
### Number of nodes
#PBS -l nodes=1:ppn=16:gpus=1
### Memory
#PBS -l mem=32gb
### Requesting time - format is <days>:<hours>:<minutes>:<seconds> (here, 12 hours)
#PBS -l walltime=10:00:00:00
 
# Go to the directory from where the job was submitted (initial directory is $HOME)
echo Working directory is $CODE/tree
cd $CODE/tree

### Here follows the user commands:
# Define number of processors
NPROCS=`wc -l < $PBS_NODEFILE`
echo This job has allocated $NPROCS nodes

# Load all required modules for the job
module purge 
source ~/.bashrc
conda activate tree-env

# This is where the work is done
# Make sure that this script is not bigger than 64kb ~ 150 lines, 
# otherwise put in separate script and execute from here
python run.py experiment=rcan_unet_sl1 only_test=true project=tree_test
