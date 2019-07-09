import sys
import matplotlib.pyplot as plt
from preprocessing import *
from pandas.plotting import autocorrelation_plot

def plot_relation_power_output(df, attr):
    df.boxplot(by=attr, column="Power average [kW]")
    plt.show()

df = read_file(sys.argv[1])
df = arrange_data(df)

#df["Wind average [m/s]"][0:100].plot()
#plt.show()
autocorrelation_plot(df["Wind average [m/s]"][0:100])
plt.show()
autocorrelation_plot(df["Wind average [m/s]"][100:200])
plt.show()
autocorrelation_plot(df["Wind average [m/s]"][200:300])
plt.show()
#plot_relation_power_output(df, "hour")
#print(df["Power average [kW]"].mean())
#print(df["Wind average [m/s]"].mean())
