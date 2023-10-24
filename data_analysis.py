#%%

import tensorboard as tb
from matplotlib import pyplot as plt
from dl_framework.experiment_data import experiment_data
import numpy as np
import os

def getmeanstd_of_data(experiment_data):

    trainacc_value_list = [] 
    testacc_value_list = []
    loss_value_list = []

    for key,tb_plot in experiment_data.data.items():
        if 'trainacc' in key:
            trainacc_value_list.append(tb_plot.value)
            steps = tb_plot.step

        if 'testacc' in key:
            testacc_value_list.append(tb_plot.value)

        if 'loss' in key:
            loss_value_list.append(tb_plot.value)

    trainacc_value_list = np.array(trainacc_value_list)
    testacc_value_list = np.array(testacc_value_list)
    loss_value_list = np.array(loss_value_list)

    meanstd_trainacc = [trainacc_value_list.mean(axis=0),trainacc_value_list.std(axis=0)] 
    meanstd_testacc = [testacc_value_list.mean(axis=0),testacc_value_list.std(axis=0)] 
    meanstd_loss = [loss_value_list.mean(axis=0),loss_value_list.std(axis=0)] 

    return meanstd_trainacc,meanstd_testacc,meanstd_loss,steps

def getkey_in_data(experiment_data,find_key):
    value_list = []
    step_list = []
    for key,tb_plot in experiment_data.data.items():
        if find_key in key:
            value_list.append(tb_plot.value)
            step_list.append(tb_plot.step)
    return value_list, step_list

def nanpad_ragged_list(data_list):
    max_len = max([len(i) for i in data_list])

    for i,data in enumerate(data_list):
        if len(data) != max_len:
            nan_array = np.empty(max_len - len(data))*np.NaN
            data_list[i] = np.append(data,nan_array)
    return np.array(data_list)

def getmeanstd_of_ad_data(experiment_data):

    auc_value_list = [] 
    loss_value_list = []
    min_len = 100000
    short_key = ''
    for key,tb_plot in experiment_data.data.items():
        if 'auc' in key:
            auc_value_list.append(tb_plot.value)
            steps = tb_plot.step
        
            if len(tb_plot.value) < min_len:
                min_len = len(tb_plot.value)  
                short_key = key

        if 'loss' in key:
            loss_value_list.append(tb_plot.value)
            
            # if len(tb_plot.value) < min_len:
            #     min_len = len(tb_plot.value)  
            #     short_key = key

    # print(f'Smallest length: {min_len} from {short_key}')

    # for i,r in enumerate(auc_value_list):
    #     auc_value_list[i] = auc_value_list[i][:min_len]
    #     loss_value_list[i] = loss_value_list[i][:min_len]
    # steps = steps[:min_len]

    steps = np.arange(max([len(i) for i in auc_value_list]))

    auc_value_list = nanpad_ragged_list(auc_value_list)
    loss_value_list = nanpad_ragged_list(loss_value_list)

    meanstd_auc = [np.nanmean(auc_value_list,axis=0),np.nanstd(auc_value_list,axis=0)] 
    meanstd_loss = [np.nanmean(loss_value_list,axis=0),np.nanstd(loss_value_list,axis=0)] 

    return meanstd_auc,meanstd_loss,steps

def get_std_interval(mean,std):
    return mean+std,mean-std

def plot_withci(meanstd_plot,name=None,steps=None,linestyle='solid',color=None):
    plt.fill_between(steps,*get_std_interval(meanstd_plot[0],meanstd_plot[1]),alpha=.2)
    plt.plot(steps,meanstd_plot[0],linestyle=linestyle,label=name)
    plt.grid()
    plt.xlabel('Epochs')

def get_data_from_folder(root):
    folder_data = []
    for i in os.listdir(root):
        print(root+i)
        folder_data.append(experiment_data(root+i))
    return folder_data

#%% Keyword Spotting Data

# All on 0.001 LR SGD??
ks_1_node = experiment_data('tb_data/ks_node/long_run_cf3be')
ks_10_node_at_10E = experiment_data('tb_data/staging/fed_vww_255ed/')
ks_10_node_at_2E = experiment_data('tb_data/fed_ks_local_epoch_sweep/fed_ks_lcl_2_c3cbf')
ks_2_node_at_10E = experiment_data('tb_data/federated/fl_vww_successful')
ks_10_node_at_2E_mom = experiment_data('tb_data/momserv/fed_ks_momfedavg/fed_ks_lcl_2_688b2')

#%% Image Classification Data

# On 0.001 LR normal SGD
cifar_1_node = experiment_data('tb_data/ic_node/long_run_1_f55b6')
cifar_2_node_at_10E = experiment_data('tb_data/cifar_210nodes_resnet/fed_ic_c244c')
cifar_10_node_at_10E = experiment_data('tb_data/fed_ic_local_epoch_sweep/fed_ic_lcl_8_df00b')
cifar_10_node_at_10E_mobilenet = experiment_data('tb_data/cifar_many_nodes/fed_ic_fcd86/')
cifar_10_node_at_2E = experiment_data('tb_data/fed_ic_local_epoch_sweep/fed_ic_lcl_2_571b3')

# Here we added LR scheduler 0.1 every 60 and upped the LR to 0.1
cifar_10_node_at_2E_mom = experiment_data('tb_data/momserv/fed_ic_momfedavg/fed_ic_lcl_2_e0eb0')

#%% Anomaly Detection Data

ad_1_node_SGD = experiment_data('tb_data/ad_node/test_run_8e7f0')

# All of these use Adam client side
ad_1_node_adam = experiment_data('tb_data/ad/ad_node_test_')
ad_2_node_at_10E = experiment_data('tb_data/ad_federated/fed_ad_success/')
ad_10_node_at_10E = experiment_data('tb_data/ad_tennodes/fed_ad_4a7d8/')
ad_10_node_at_2E = experiment_data('tb_data/fed_ad_local_epoch_sweep/fed_ad_lcl_2_9ee82')
ad_10_node_at_2E_mom = experiment_data('tb_data/momserv/fed_ad_momfedavg/fed_ad_lcl_2_f9bcf')

#%% Visual Wakewords

vww_1_data = experiment_data('tb_data/old/mbv2_vww_rcrop')
vww_10_data = experiment_data('tb_data/staging/fed_vww_255ed/')
vww_2_data = experiment_data('tb_data/federated/fl_vww_successful/')

#%%

ks_list = [ks_1_node,ks_10_node_at_10E,ks_10_node_at_2E,ks_2_node_at_10E,ks_10_node_at_2E_mom]
ks_names = ['KS Baseline',
            'KS 10 Nodes @ E=10',
            'KS 10 Nodes @ E=2',
            'KS 2 Nodes @ E=10',
            'KS 10 nodes @ E=2 w/ server momentum']

ks_linestyles = ['--',
            ':',
            ':',
            '--',
            '-']

ks_meanstd = [getmeanstd_of_data(i) for i in ks_list]


for i,meanstd in enumerate(ks_meanstd):
    plot_withci(meanstd[1],name=ks_names[i],steps=meanstd[3],linestyle=ks_linestyles[i])
steps = meanstd[3]
plt.plot(steps,[0.9]*len(steps),c='black')
plt.legend()
plt.ylim(0.6,0.95)
plt.xlim(-5,100)
plt.title('Federated MLPerfTiny Keyword Spotting')
plt.ylabel('Accuracy')
plt.grid()
plt.grid()
plt.show()

for i,meanstd in enumerate(ks_meanstd):
    print(f'Max {ks_names[i]} : {np.max(meanstd[1]).round(3)}')

baseline = np.max(ks_meanstd[0][1])
for i,meanstd in enumerate(ks_meanstd):
    print(f'Max {ks_names[i]} : {(np.max(meanstd[1])-0.9).round(3)}')

#%%

ic_list = [cifar_1_node,
           cifar_2_node_at_10E,
           cifar_10_node_at_10E,
           #cifar_10_node_at_10E_mobilenet,
           cifar_10_node_at_2E,
           cifar_10_node_at_2E_mom]

ic_names =['IC Baseline',
           'IC 2 Nodes @ E=10',
           'IC 10 Nodes @ E=8',
           #'IC 10 Nodes @ E=10, with mobilenet',
           'IC 10 Nodes @ E=2',
           'IC 10 Nodes @ E=10, with momentum']

ic_meanstd = [getmeanstd_of_data(i) for i in ic_list]

for i,meanstd in enumerate(ic_meanstd):
    # if i == 3:
    #     plot_withci(meanstd[1],name=ic_names[i],steps=meanstd[3],linestyle='dotted')
    #     continue
    if i in [0,1]:
        plot_withci(meanstd[1],name=ic_names[i],steps=meanstd[3],linestyle='dashed')
        continue
    if i in [2,3]:
        plot_withci(meanstd[1],name=ic_names[i],steps=meanstd[3],linestyle='dotted')
        continue
    plot_withci(meanstd[1],name=ic_names[i],steps=meanstd[3])

steps = meanstd[3]
plt.plot(steps,[0.85]*len(steps),c='black')
plt.legend()
plt.ylim(0.3,0.95)
plt.xlim(-5,200)
plt.title('Federated MLPerfTiny IC')
plt.ylabel('Accuracy')
plt.grid()
plt.grid()
plt.show()

for i,meanstd in enumerate(ic_meanstd):
    print(f'Max {ic_names[i]} : {np.max(meanstd[1]).round(3)}')

baseline = np.max(ic_meanstd[0][1])
for i,meanstd in enumerate(ic_meanstd):
    print(f'Max {ic_names[i]} : {(np.max(meanstd[1])-0.9).round(3)}')

#%%

vww_list = [vww_1_data,
           vww_2_data,
           vww_10_data]

vww_names =['VWW Baseline',
           'VWW 2 Nodes @ E=10',
           'VWW 10 Nodes @ E=10']

vww_meanstd = [getmeanstd_of_data(i) for i in vww_list]

for i,meanstd in enumerate(vww_meanstd):
    if i == 2:
        plot_withci(meanstd[1],name=vww_names[i],steps=meanstd[3],linestyle=':')
        continue
    plot_withci(meanstd[1],name=vww_names[i],steps=meanstd[3])

steps = meanstd[3]
plt.plot(steps,[0.8]*len(steps),c='black')
plt.legend()
plt.ylim(0.6,0.95)
plt.xlim(-5,100)
plt.title('Federated MLPerfTiny VWW')
plt.ylabel('Accuracy')
plt.grid()
plt.grid()
plt.show()

for i,meanstd in enumerate(vww_meanstd):
    print(f'Max {vww_names[i]} : {np.max(meanstd[1]).round(3)}')

baseline = np.max(vww_meanstd[0][1])
for i,meanstd in enumerate(vww_meanstd):
    print(f'Max {vww_names[i]} : {(np.max(meanstd[1])-0.9).round(3)}')

#%%

ad_list = [ad_1_node_SGD,
        ad_1_node_adam,
        ad_2_node_at_10E,
        ad_10_node_at_8E,
        ad_10_node_at_2E,
        ad_10_node_at_2E_mom]

ad_names = ['AD Baseline with SGD',
            'AD Baseline with Adam',
            'AD 2 Nodes @ E=10',
            'AD 10 Nodes @ E=8',
            'AD 10 Nodes @ E=2',
            'AD 10 Nodes @ E=2, server momentum']

ad_linestyles = ['--',
                 '--',
                 ':',
                 ':',
                 ':',
                 '-']

ad_meanstd = [getmeanstd_of_ad_data(i) for i in ad_list]

for i,meanstd in enumerate(ad_meanstd):
    plot_withci(meanstd[0],name=ad_names[i],steps=meanstd[2],linestyle=ad_linestyles[i])
steps = np.arange(184)
plt.plot(steps,[0.85]*len(steps),c='black')
plt.ylim(0.7,0.9)
plt.xlim(-5,184)
plt.title('Federated MLPerfTiny AD')
plt.ylabel('AUC')
plt.legend()
plt.grid()

for i,meanstd in enumerate(ad_meanstd):
    print(f'Max {ad_names[i]} : {np.max(meanstd[0]).round(3)}')

baseline = np.max(ad_meanstd[0][1])
for i,meanstd in enumerate(ad_meanstd):
    print(f'Max {ad_names[i]} : {(np.max(meanstd[0])-0.9).round(3)}')


#%%

ad_10_node_at_8E = experiment_data('tb_data/fed_ad_local_epoch_sweep/fed_ad_lcl_8_2f9f1')

#%%

ad_1_node_SGD = experiment_data('tb_data/ad_node_sgd/sgd_run_76434')