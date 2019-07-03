import sys
import matplotlib.pyplot as plt
from preprocessing import *

def plot_relation_power_output(df, attr):
    df.boxplot(by=attr, column="Power average [kW]")
    plt.show()

df = read_file(sys.argv[1])
df = arrange_data(df)

#plot_relation_power_output(df, "hour")
print(df["Power average [kW]"].mean())
print(df["Wind average [m/s]"].mean())