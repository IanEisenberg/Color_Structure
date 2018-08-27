scripts_loc=~/Experiments/Prob_Context_Task/fmri/analysis
data_loc=/mnt/OAK/prob_context/BIDS_data    

# as script
docker run --rm  \
--mount type=bind,src=$scripts_loc,dst=/scripts \
--mount type=bind,src=$data_loc,dst=/data \
-ti fmri_env \
python task_analysis.py /output /Data --participant s358 --tasks stopSignal 

# as notebook
docker run --rm  \
--mount type=bind,src=$scripts_loc,dst=/scripts \
--mount type=bind,src=$data_loc,dst=/data \
-ti -p 8888:8888 fmri_env \