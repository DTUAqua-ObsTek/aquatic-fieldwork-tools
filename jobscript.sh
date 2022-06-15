#!/bin/sh
### General options
### - specify queue --
#BSUB -q hpc
### -- set the job Name --
#BSUB -J videoprocessing
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 16GB of system-memory
#BSUB -R "rusage[mem=16GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o videoproc_%J.out
#BSUB -e videoproc_%J.err
# -- end of LSF options --
source $HOME/aquatic-fieldwork-tools/venv/bin/activate
cd $HOME/aquatic-fieldwork
echo "Starting processing job."
python video_proc.py --configuration configuration.ini $HOME/videos
echo "Processing finished"