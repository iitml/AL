"""
The :mod:`utils.utils` implements various helper functions.
"""
import matplotlib.pyplot as plt

def data_to_file(filename, strategy, accu_y, auc_y, values):
    if filename:
        f = open(filename, 'a')
    else:
        f = open('avg_results.txt', 'a')

    #Write Accuracy Plot Values
    f.write(strategy+'\n'+'accuracy'+'\n')
    f.write('train size,mean'+'\n')
    for i in range(len(accu_y)):
        f.write("%d,%f\n" % (values[i], accu_y[i]))
    f.write('\n')

    #Write AUC Plot Values
    f.write('AUC'+'\n')
    f.write('train size,mean'+'\n')
    for i in range(len(auc_y)):
        f.write("%d,%f\n" % (values[i], auc_y[i]))
    f.write('\n\n\n')

    f.close()

def assign_plot_params(avg_accu, avg_auc):
    # Accuracy Plot Values
    accu_x = sorted(avg_accu.keys())
    accu_y = [avg_accu[xi] for xi in accu_x]

    # AUC Plot Values
    auc_x = sorted(avg_auc.keys())
    auc_y = [avg_auc[xi] for xi in auc_x]

    return accu_x, accu_y, auc_x, auc_y

def draw_plots(strategy, accu_x, accu_y, auc_x, auc_y):
    plt.figure(1)
    plt.subplot(211)
    plt.plot(accu_x, accu_y, '-', label=strategy)
    plt.legend(loc='best')
    plt.title('Accuracy')

    plt.subplot(212)
    plt.plot(auc_x, auc_y, '-', label=strategy)
    plt.legend(loc='best')
    plt.title('AUC')

def show_plt():
    plt.show()
