"""
The :mod:`utils.utils` implements various helper functions.
"""
import matplotlib.pyplot as plt

def data_to_file(filename, strategy, accu_y, auc_y, values):
    """Places data in file

    **Parameters**

    * filename (*str*) - user-specified path
    * strategy
    * accu_y (*list*)
    * auc_y (*list*)
    * values (*list*)

    """
    if filename and filename != "''":
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

def data_to_py(filename, c, st, acc_x, acc_y, auc_x, auc_y):

    """Places plot data in python file

    **Parameters**

    * filename (*str*) - user-specified path
    * c - classifier
    * st - strategy
    * acc_x (*list*)
    * accu_y (*list*)
    * auc_x (*list*)
    * auc_y (*list*)

    """
    plot_valsf = open(filename, 'a')
    plot_valsf.write('vals["%s_%s_accx"]=%s\n' % (c, st, str(acc_x)))
    plot_valsf.write('vals["%s_%s_accy"]=%s\n' % (c, st, str(acc_y)))
    plot_valsf.close()

    plot_valsf = open(filename, 'a')
    plot_valsf.write('vals["%s_%s_aucx"]=%s\n' % (c, st, str(auc_x)))
    plot_valsf.write('vals["%s_%s_aucy"]=%s\n' % (c, st, str(auc_y)))
    plot_valsf.close()

def assign_plot_params(avg_accu, avg_auc):
    """Assigns plot parameters

    **Parameters**

    * avg_accu - respective average accuracy performance
    * avg_auc - respective average auc performance

    **Returns**

    * accu_x (*list*)
    * accu_y (*list*)
    * auc_x (*list*)
    * auc_y (*list*)

    """
    # Accuracy Plot Values
    accu_x = sorted(avg_accu.keys())
    accu_y = [avg_accu[xi] for xi in accu_x]

    # AUC Plot Values
    auc_x = sorted(avg_auc.keys())
    auc_y = [avg_auc[xi] for xi in auc_x]

    return accu_x, accu_y, auc_x, auc_y

def draw_plots(strategy, accu_x, accu_y, auc_x, auc_y):
    """Draws the plot

    **Parameters**

    * strategy
    * accu_x (*list*)
    * accu_y (*list*)
    * auc_x (*list*)
    * auc_y (*list*)

    """
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
    """Shows the plot"""
    plt.show()
