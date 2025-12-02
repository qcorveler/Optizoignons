import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def trace_cap_util(df:pd.DataFrame, points:list=None, title:str="Courbe d'utlisation de la capacité"):
    
    fig, ax = plt.subplots(figsize=(10,6))

    ax.plot(df.index,df, color='grey', 
            linestyle='-', linewidth=1, )
    ax.set_xlabel('selling_periods', size=11)
    # ax.set_xticks(np.arange(min(dft.index), max(dft.index)+1, 1.0))
    ax.set_ylabel("capacity utilization" , size=11)

    y_majorLocator = MultipleLocator(0.1)
    ax.yaxis.set_major_locator(y_majorLocator)
    x_majorLocator = MultipleLocator(10)
    ax.xaxis.set_major_locator(x_majorLocator)

    ax.set_title(title, size=12)
    # ax.legend(loc=0)
    if points is not None:
        print("Points to plot:", points)    
        for point in points:
            print(point)
            plt.scatter(point[0], point[1])
    ax.grid(True)
    fig.tight_layout()

    plt.show()
