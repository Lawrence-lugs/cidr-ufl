#%% 

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import matplotlib.pyplot as plt

class tb_plot(object):
    def __init__(self,name,value,wall_time,step):
        self.value = value
        self.wall_time = wall_time
        self.step = step
        self.name = name

    def plot(self):
        plt.plot(self.step,self.value)
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.title(f'{self.name}')

class experiment_data(object):
    def __init__(self, tb_datafolder_path : str):
        event_acc = EventAccumulator(tb_datafolder_path)
        event_acc.Reload()
        plot_name_list = event_acc.Tags()['scalars'] 
         
        self.data = {}

        for plot_name in plot_name_list:
            value = np.array([i.value for i in event_acc.Scalars(plot_name)])
            wall_time = np.array([i.wall_time for i in event_acc.Scalars(plot_name)])
            step = np.array([i.step for i in event_acc.Scalars(plot_name)])

            self.data[plot_name] = tb_plot(plot_name,value,wall_time,step)

if __name__ == '__main__':
    my_data = experiment_data('tb_data/staging/fed_vww_255ed/')

#%%

    import matplotlib.pyplot as plt

    my_data.data['data/node_4/loss'].plot()

