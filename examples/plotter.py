import matplotlib.pyplot as plt
import numpy as np

class Colors:
    colors = [
        "#e74c3c",
        "#1abc9c",
        "#3498db",
        "#8e44ad",
        "#f1c40f",
        "#34495e",
        "#95a5a6",
    ]

class Confidence_Z:
    z = {
        0.9: 1.65,
        0.95: 1.96,
        0.99: 2.58
        }


class Plotter:

    def __init__(self, fontsize=12):
        """
        self.index: used to control which color
        """
        # set all font size to 10
        plt.rc('font', size=fontsize)
        self.axis_linewdith = 2
        self.conf_int_alpha = 0.2

        # index
        self.index = 0

    def reset(self):
        self.index = 0

    def compute_mean_and_conf_int(self, y, compute_conf_int=True, conf_level=0.95):
        """
        Compute mean and/or confidence interval for y for a given confidence level

        Returns mean of data, confidence interval low, confidence interval high
        """
        y = np.array(y)
        if y.ndim == 1 or y.shape[0] == 1:
            # one trial/seed
            # cannot compute conf int
            y = y.reshape(1, -1)
            compute_conf_int = False

        # mean
        y_mean = np.mean(y, axis=0)
        if not compute_conf_int:
            return y_mean, None, None
        
        ### compute conf int

        # sample std
        y_sample_std = np.std(y, axis=0, ddof=1)
        z = Confidence_Z.z[conf_level]
        n = y.shape[0]
        conf_int_low = y_mean - z * y_sample_std / np.sqrt(n)
        conf_int_high = y_mean + z * y_sample_std / np.sqrt(n)

        return y_mean, conf_int_low, conf_int_high

    def plot_data(self, ax, x, y, conf_level=0.95, plot_conf_int=True, plot_all_seeds=False):
        """
        x: x axis data of shape (1, d) or (d,)
        y: y axis data of shape (n, d), where n is number of trials/seeds, and d is data length
        """
        x = np.array(x)
        y = np.array(y)
        # cannot do both at the same time
        assert not (plot_conf_int == True and plot_all_seeds == True)

        color = Colors.colors[self.index] 

        # plot data
        y_mean, conf_int_low, conf_int_high = self.compute_mean_and_conf_int(y, compute_conf_int=plot_conf_int, conf_level=conf_level)
        ax.plot(x, y_mean, color=color)

        # plot confidence interval
        if plot_conf_int:
            ax.fill_between(x, conf_int_low, conf_int_high, alpha=self.conf_int_alpha, color=color)

        # plot all seeds
        if plot_all_seeds:
            for y_seed in y:
                ax.plot(x, y_seed, color=color, alpha=self.conf_int_alpha)


        # # text label of curve
    def draw_text(self, ax, x_pos, y_pos, text):
        ax.text(x_pos, y_pos, text, color=Colors.colors[self.index])


    def spine_setting(self, ax):
        # axis spine settings
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines["left"].set_linewidth(self.axis_linewdith)
        ax.spines["bottom"].set_linewidth(self.axis_linewdith)

    def axis_setting(self, ax, xlim_low=0, xlim_high=None, ylim_low=None, ylim_high=None):
        ax.set_xlim([xlim_low, xlim_high])
        ax.set_ylim([ylim_low, ylim_high])

    def finalize_plot(self, ax, title=None, xlabel=None, ylabel=None, axis_setting=None):
        """
        axis_setting: list, [xlim low, xlim high, ylim low, ylim high]
        """
        # axis spine setting
        self.spine_setting(ax)

        # axis setting
        if axis_setting is None:
            self.axis_setting(ax)
        else:
            self.axis_setting(ax, *axis_setting)

        # texts
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)

        plt.tight_layout()

        
        
    


if __name__ == "__main__":
    plotter = Plotter()
     

    fig, ax = plt.subplots(figsize=(10, 5))

    for i in range(5):
        x = np.arange(200)
        y = np.random.normal(0, 5, size=(5, 200))  

        plotter.plot_data(ax, x, y, plot_all_seeds=False, plot_conf_int=True)
        plotter.index += 1
        
    plotter.finalize_plot(ax, title="Dayan", xlabel="Timesteps", y_label="Return")
    plt.show()


    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    plotter.reset()
    for ax in axes.flatten():
        x = np.arange(200)
        y = np.random.normal(0, 5, size=(5, 200))  
    
        plotter.plot_data(ax, x, y)


        plotter.finalize_plot(ax, xlabel="Timesteps", y_label="Return")


    plt.show()