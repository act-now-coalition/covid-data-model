import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_results(results):
    labels = {'net_cost': '(test cost - avoided cost) per month ($)',
              'delta_p_infected': 'reduced percentage of infection',
              'delta_outbreak_prob': 'reduced probability of outbreak\n (worksite closure)',
              'delta_covid_index': 'reduced covid index'}

    for variable in labels:
        ax = plt.figure(figsize=(12, 6)).gca(projection='3d')
        img = ax.scatter(results['pcr_coverage'], results['pcr_frequency'] * 30, results['antibody_coverage'],
                         c=results[variable], cmap=plt.hot())
        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes('right', size='50%', pad=0.05)
        plt.colorbar(img, label=labels[variable], orientation='vertical')
        ax.set_xlabel('pcr coverage (%)')
        ax.set_ylabel('pcr frequency (per month)')
        ax.set_zlabel('antibody_coverage (%)')
        plt.title(f'effect of testing strategy on {labels[variable]}')
        plt.tight_layout()
        plt.show()