import matplotlib.pyplot as plt
from Analysis.plot_utils import plot_threshold_run
from Analysis.load_data import load_threshold_data

subjid = input('Enter Subject ID: ')
N = input('How many of the previous trials should be used?: ')
N = None if N == '' else int(N)
taskdata, motion_df = load_threshold_data(subjid, dim='motion')
taskdata, ori_df = load_threshold_data(subjid, dim='orientation')
f = plot_threshold_run(subjid, N=N)
plt.show()