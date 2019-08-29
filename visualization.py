import sys
import matplotlib.pyplot as plt
from preprocessing import *
from pandas.plotting import autocorrelation_plot
import numpy as np

def plot_predicted_vs_truth(predicted, truth, miny, maxy, name=None):
    plt.plot([x for x in range(len(predicted))], predicted, label='predicted')
    plt.plot([x for x in range(len(predicted))], truth, label='truth')
    axes = plt.gca()
    axes.set_ylim([miny, maxy])
    plt.legend()
    if name is not None:
        plt.savefig(name)
    else:
        plt.show()
    plt.close()

def plot_relation_power_output(df, attr):
    df.boxplot(by=attr, column="Power average [kW]", showfliers=False)
    axes = plt.gca()
    plt.show()
    plt.close()

#df["Wind average [m/s]"][0:100].plot()
#plt.show()
if __name__ == "__main__":
    df = read_file(sys.argv[1])
    df = arrange_data(df)

    plot_relation_power_output(df, "hour")

#print(df["Power average [kW]"].mean())
#print(df["Wind average [m/s]"].mean())
