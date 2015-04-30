"""
The GUI module to run the active learning strategies.
"""
import os, sys
path = os.path.join(os.path.dirname("__file__"), '../..')
sys.path.insert(0, path)

from __init__ import *
from al.learning_curve import LearningCurve
from utils.utils import *
import plot_vals

'''Values'''
plot_col = 885
plotB_col = 850
class_col = 400
strat_col = 600
show_col = 500
run_col = 165
run_class_col = 65
run_strat_col = 265
label_limit = 35

run_params = {}
show_params = {}

show_params = {}
show_params_clas = {}
show_params_strat = {}
run_params = {}
clas_params = {}
clas_params_clas = {}
strat_params = {}
strat_params_strat = {}

class HelperFunctions(object):
    """Class - includes helper functions."""
    def __init__(self):
        """Instantiate loaded data paths"""
        self.train_load_val = StringVar()
        self.test_load_val = StringVar()
        self.single_load_val = StringVar()


    def all_combos(self):
        """Retrieve all possible combinations of classifier and strategy"""
        result = []
        for clas in clas_params_clas:
          for strat in strat_params_strat:
            clas_name = re.findall('(\w+)CheckVal_run', clas)[0]
            strat_name = re.findall('(\w+)CheckVal_run', strat)[0]
            result.append((clas_name, strat_name))
        return result


    def load_data(self, dataset1, dataset2=None):
        """Loads the dataset(s) given in the the svmlight / libsvm format
        and assumes a train/test split

        **Parameters**

        * dataset1 (*str*) - Path to the file of the first dataset.
        * dataset2 (*str or None*) - If not None, path to the file of second dataset

        **Returns**

        * Pool and test files - X_pool, X_test, y_pool, y_test

        """
        if dataset2:
            X_pool, y_pool = load_svmlight_file(dataset1)
            _, num_feat = X_pool.shape

            # Splitting 2/3 of data as training data and 1/3 as testing
            # Data selected randomly
            X_test, y_test = load_svmlight_file(dataset2, n_features=num_feat)

        else:
            X, y = load_svmlight_file(dataset1)
            X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=(1./3.), random_state=42)

        return (X_pool, X_test, y_pool, y_test)

    def open_data(self, filetype):
        """Set label values in gui and call :mod:`front_end.gui.run_al_gui.HelperFunctions.load_data` and :mod:`front_end.gui.run_al_gui.HelperFunctions.gray_run`

        **Parameters**

        * filetype (*str*) - 'train', 'test', or 'single'

        """
        data_f = tkFileDialog.askopenfilename()
        if data_f == ():
          if filetype == 'train':
            self.train_load_val.set("''")

          elif filetype == 'test':
            self.test_load_val.set("''")

          else:
            self.single_load_val.set("''")
        else:
          if filetype=='train':
            self.train_load_val.set(data_f)
          elif filetype=='test':
            self.test_load_val.set(data_f)
          else:
            self.single_load_val.set(data_f)

        if self.single_load_val.get() != "''":
            self.X_pool, self.X_test, self.y_pool, self.y_test = self.load_data(self.single_load_val.get())

        elif self.test_load_val.get() != "''" and self.train_load_val.get() != "''":
            self.X_pool, self.X_test, self.y_pool, self.y_test = self.load_data(self.train_load_val.get(), self.test_load_val.get())

        self.gray_run()

    def gray_run(self):
        """Enables or disables run checkboxes depending on if the data has been loaded"""
        if (self.train_load_val.get() != "''" and self.test_load_val.get() != "''") or self.single_load_val.get() != "''":
          for itm in strat_params:
            strat_params[itm].config(state=NORMAL)

          for itm in clas_params:
            clas_params[itm].config(state=NORMAL)

        else:
          for itm in strat_params:
            strat_params[itm].config(state=DISABLED)

          for itm in clas_params:
            clas_params[itm].config(state=DISABLED)

    def gray_out(self):
        """Enables or disables the show_plots checkboxes depending on which classifiers - strategies have been run"""
        for itm in show_params:
          show_params[itm].config(state=DISABLED)

        run_list = open('files/run_list.txt', 'r')
        run_list_r = run_list.read()
        run_list.close()

        clas_strat_all = self.all_combos()
        for clas_name, strat_name in clas_strat_all:
          if "%s-%s" % (clas_name, strat_name) in run_list_r:
            show_params["%sCheckBox_2" % clas_name].config(state=NORMAL)
            show_params["%sCheckBox_2" % strat_name].config(state=NORMAL)


class ParamsWindow(object):
    """Class - shows the parameters window in edit->parameters"""
    def __init__(self):
        '''Instantiates :mod:`front_end.gui.run_al_gui.ParamsWindow` and calls :mod:`front_end.gui.run_al_gui.ParamsWindow.display_params`'''
        self.pref = Toplevel(takefocus=True)
        self.pref.title("Parameters")
        self.pref.geometry("200x450+110+100")
        self.pref.withdraw()

        self.pref.protocol('WM_DELETE_WINDOW', self.exit_pref)

        self.pref.attributes("-alpha", 1.0)
        self.pref_canvas = Canvas(self.pref)
        self.pref_canvas.pack(fill=BOTH, expand="true")
        self.display_params()

    def display_pref(self):
        """Display the parameters in a separate window"""
        self.pref.deiconify()

    def exit_pref(self):
        """Close the parameters window"""
        self.pref.withdraw()

    def check_int(self, param):
        """Check to make sure the user-defined parameter is a valid integer

        **Parameters**

        * param (*tuple*) - user-defined parameter

        **Returns**

        True or false - depends on if the parameter is or is not a valid integer respectively

        """
        try:
          int(param[0].get())
          return True
        except ValueError:
          tkMessageBox.showinfo("Error", "%s value is not a number!\nSetting Default" % param[2])
          param[0].set(param[1])
          self.pref.attributes('-topmost', 1)
          return False

    def display_params(self):
        """Create labels, entry boxes, etc. for the parameters window"""
        self.paramTitle = Label(self.pref_canvas, text="Parameters")
        self.label_window = self.pref_canvas.create_window(0, 8, anchor = NW, window=self.paramTitle)

        run_params["bs_val"] = (StringVar(), "10", "Bootstrap")
        self.bs_label = Label(self.pref_canvas, text=run_params["bs_val"][2])
        self.bs_label_window = self.pref_canvas.create_window(60, 40, anchor=NW, window=self.bs_label)

        run_params["bs_val"][0].set(run_params["bs_val"][1])
        self.bs_box = Entry(self.pref_canvas, textvariable = run_params["bs_val"][0], bd=5, validatecommand=lambda: self.check_int(run_params["bs_val"]), validate="focusout")
        self.bs_box_window = self.pref_canvas.create_window(5, 60, anchor=NW, window=self.bs_box)

        run_params["sz_val"] = (StringVar(), "10", "Step Size")
        self.sz_label = Label(self.pref_canvas, text=run_params["sz_val"][2])
        self.sz_label_window = self.pref_canvas.create_window(60, 100, anchor=NW, window=self.sz_label)

        run_params["sz_val"][0].set(run_params["sz_val"][1])
        self.sz_box = Entry(self.pref_canvas, textvariable =run_params["sz_val"][0], bd=5, validatecommand=lambda: self.check_int(run_params["sz_val"]), validate="focusout")
        self.sz_box_window = self.pref_canvas.create_window(5, 120, anchor=NW, window=self.sz_box)

        run_params["b_val"] = (StringVar(), "500", "Budget")
        self.b_label = Label(self.pref_canvas, text=run_params["b_val"][2])
        self.b_label_window = self.pref_canvas.create_window(60, 160, anchor=NW, window=self.b_label)

        run_params["b_val"][0].set(run_params["b_val"][1])
        self.b_box = Entry(self.pref_canvas, textvariable =run_params["b_val"][0], bd=5, validatecommand=lambda: self.check_int(run_params["b_val"]), validate="focusout")
        self.b_box_window = self.pref_canvas.create_window(5, 180, anchor=NW, window=self.b_box)

        run_params["sp_val"] = (StringVar(), "250", "Subpool")
        self.sp_label = Label(self.pref_canvas, text=run_params["sp_val"][2])
        self.sp_label_window = self.pref_canvas.create_window(60, 220, anchor=NW, window=self.sp_label)

        run_params["sp_val"][0].set(run_params["sp_val"][1])
        self.sp_box = Entry(self.pref_canvas, textvariable =run_params["sp_val"][0], bd=5, validatecommand=lambda: self.check_int(run_params["sp_val"]), validate="focus")
        self.sp_box_window = self.pref_canvas.create_window(5, 240, anchor=NW, window=self.sp_box)

        run_params["nt_val"] = (StringVar(), "10", "Number of Trials")
        self.nt_label = Label(self.pref_canvas, text=run_params["nt_val"][2])
        self.nt_label_window = self.pref_canvas.create_window(35, 280, anchor=NW, window=self.nt_label)

        run_params["nt_val"][0].set(run_params["nt_val"][1])
        self.nt_box = Entry(self.pref_canvas, textvariable = run_params["nt_val"][0], bd=5, validatecommand=lambda: self.check_int(run_params["nt_val"]), validate="focusout")
        self.nt_box_window = self.pref_canvas.create_window(5, 300, anchor=NW, window=self.nt_box)

        self.file_label = Label(self.pref_canvas, text="Filename")
        self.pref_canvas.create_window(60, 340, anchor=NW, window=self.file_label)

        self.file_inputvar = StringVar()
        self.file_inputvar.set("''")
        self.file_input = Entry(self.pref_canvas, textvariable=self.file_inputvar, bd=5)
        self.pref_canvas.create_window(5, 360, anchor=NW, window=self.file_input)

        self.setButton = Button(self.pref_canvas, text="Set", command=self.exit_pref)
        self.pref_canvas.create_window(65, 400, anchor=NW, window=self.setButton)


class MenuWindow(object):
    """Class - creates the menu bar (including the file and edit menus)"""
    def __init__(self, master):
        """Instantiates :mod:`front_end.gui.run_al_gui.MenuWindow; calls :mod:`front_end.gui.run_al_gui.MenuWindow.show_filemenu` and :mod:`front_end.gui.run_al_gui.MenuWindow.show_editmenu`

        **Parameters**

        * master - main Tkinter window

        """

        self.menubar = Menu(master)
        self.pref_w = ParamsWindow()
        self.helper = HelperFunctions()

        self.show_filemenu(master)
        self.show_editmenu(master)

    def show_filemenu(self, master):
        """Creates the file menubar and calls :mod:`front_end.gui.run_al_gui.HelperFunctions.open_data`

        **Parameters**

        * master - main Tkinter window

        """

        self.filemenu = Menu(self.menubar, tearoff=0)
        self.filemenu.add_command(label="Load single dataset", command=lambda: self.helper.open_data('single'))
        self.filemenu.add_command(label="Load train dataset", command=lambda: self.helper.open_data('train'))
        self.filemenu.add_command(label="Load test dataset", command=lambda: self.helper.open_data('test'))
        self.filemenu.add_command(label="Quit           (ESC)", command=master.quit)
        self.menubar.add_cascade(label="File", menu=self.filemenu)

    def show_editmenu(self, master):
        """Creates the edit menubar and calls :mod:`front_end.gui.run_al_gui.ParamsWindow.display_pref`

        **Parameters**

        * master - main Tkinter window

        """
        self.editmenu = Menu(self.menubar, tearoff=0)
        self.editmenu.add_command(label="Parameters", command=self.pref_w.display_pref)
        self.menubar.add_cascade(label="Edit", menu=self.editmenu)

class MainCanvas(object):
    """Class - creates main canvas (window)"""
    def __init__(self, master):
        """Instantiates :mod:`front_end.gui.run_al_gui.MainCanvas; calls :mod:`front_end.gui.run_al_gui.MainCanvas.add_classifier_frame_2`, :mod:`front_end.gui.run_al_gui.add_strategy_frame_2`, :mod:`front_end.gui.run_al_gui.add_run_classifier_frame`, :mod:`front_end.gui.run_al_gui.add_strategy_frame`, :mod:`front_end.gui.run_al_gui.add_buttons`, :mod:`front_end.gui.run_al_gui.add_alerts`

        Creates the file menubar and calls :mod:`front_end.gui.run_al_gui.HelperFunctions.open_data`

        **Parameters**

        * master - main Tkinter window

        """
        self.menu_w = MenuWindow(master)

        self.w = Canvas(master)
        self.w.pack(fill=BOTH, expand="true")
        self.times_i_10 = tkFont.Font(root=master, family="Times New Roman", slant="italic", size=11)

        self.back_image = Image.open("img/background.jpg")
        self.background_image = ImageTk.PhotoImage(self.back_image)
        self.background_label = Label(image=self.background_image)
        self.w.create_window(0, 0, anchor = NW, window=self.background_label)

        self.showMessage_Label = Label(text="Plot The Following", font=self.times_i_10, bg="grey")
        self.w.create_window(show_col, 8, anchor=NW, window=self.showMessage_Label)

        self.runMessage_Label = Label(text="Run Settings", font=self.times_i_10, bg="grey")
        self.w.create_window(run_col, 300, anchor=NW, window=self.runMessage_Label)

        self.classifierLabel = Label(text="Classifiers", font=self.times_i_10, bg="grey")
        self.classifierLabel_window = self.w.create_window(class_col, 60, anchor=NW, window=self.classifierLabel)

        self.strategyLabel = Label(text="Strategies", font=self.times_i_10, bg="grey")
        self.strategyLabel_window = self.w.create_window(strat_col, 60, anchor=NW, window=self.strategyLabel)

        self.add_classifier_frame_2(master)
        self.add_strategy_frame_2(master)
        self.add_run_classifier_frame(master)
        self.add_run_strategy_frame(master)
        self.add_buttons()
        self.add_alerts()

    def plot_acc(self, clas_strat, width_org, height_org, savefile):
        """Plots accuracy

        **Parameters**

        clas_strat (*list*) - classifier-strategy combinations
        width_org (*int*) - picture width
        height_org (*int*) - picture height
        savefile (*str*) - path to picture's save location

        """
        plt.clf()
        try:
          for item in clas_strat:
            reload(plot_vals)
            plt.plot(plot_vals.vals["%s_%s_accx" % (item[0], item[1])], plot_vals.vals["%s_%s_accy" % (item[0], item[1])], '-', label='%s_%s' % (item[0], item[1]))
            plt.legend(loc='best')
            plt.title('Accuracy')
            plt.savefig('img/plot_acc.png')
            if savefile:
              plt.savefig(savefile)

          filename = "img/plot_acc.png"
          accuracyPlot = Image.open(filename).resize((width_org, height_org), Image.ANTIALIAS)
        except:
          filename = "img/blank_acc.png"
          accuracyPlot = Image.open(filename).resize((width_org, height_org), Image.ANTIALIAS)
        accuracyPlot_image = ImageTk.PhotoImage(accuracyPlot)
        accuracyPlot_label = Label(image=accuracyPlot_image)
        accuracyPlot_label.image = accuracyPlot_image
        self.w.create_window(plot_col-180, 15, anchor = NW, window=accuracyPlot_label)

    def plot_auc(self, clas_strat, width_org, height_org, savefile):
        """Plots auc

         **Parameters**

        clas_strat (*list*) - classifier-strategy combinations
        width_org (*int*) - picture width
        height_org (*int*) - picture height
        savefile (*str*) - path to picture's save location

        """
        plt.clf()
        try:
          for item in clas_strat:
            reload(plot_vals)
            plt.plot(plot_vals.vals["%s_%s_aucx" % (item[0], item[1])], plot_vals.vals["%s_%s_aucy" % (item[0], item[1])], '-', label='%s_%s' % (item[0], item[1]))
            plt.legend(loc='best')
            plt.title('AUC')
            plt.savefig('img/plot_auc.png')
            if savefile:
              plt.savefig(savefile)

          filename = "img/plot_auc.png"
          img_org = Image.open(filename)
        except Exception, e:
          print str(e)
          filename = "img/blank_auc.png"
          img_org = Image.open(filename)

        aucPlot = img_org.resize((width_org, height_org), Image.ANTIALIAS)
        aucPlot_image = ImageTk.PhotoImage(aucPlot)
        aucPlot_label = Label(image=aucPlot_image)
        aucPlot_label.image = aucPlot_image
        aucPlot_w = self.w.create_window(plot_col-180, 305, anchor = NW, window=aucPlot_label)

    def show_plots(self, auc_save=False, acc_save=False):
        """Show the plots; calls :mod:`front_end.gui.run_al_gui.MainCanvas.plot_acc` and :mod:`front_end.gui.run_al_gui.MainCanvas.plot_auc`

        **Parameters**

        * auc_save (*str*) - False or path to auc plot's save location
        * acc_save (*str*) - False or path to accuracy plot's save location

        """
        width_org, height_org = (380, 280)
        clas_strat = []
        for clas in show_params_clas:
          for strat in show_params_strat:
            if show_params_clas[clas].get() and show_params_strat[strat].get():
              clas_name = re.findall('(\w+)CheckVal', clas)[0]
              strat_name = re.findall('(\w+)CheckVal', strat)[0]
              clas_strat.append((clas_name, strat_name))

        self.plot_acc(clas_strat, width_org, height_org, acc_save)
        self.plot_auc(clas_strat, width_org, height_org, auc_save)


    def clear_plots(self):
        """Clear the plots and show empty plots; calls :mod:`front_end.gui.run_al_gui.MainCanvas.clean` and :mod:`front_end.gui.run_al_gui.MainCanvas.show_plots`"""
        self.clean(show_params_clas)
        self.clean(show_params_strat)
        self.show_plots()

    def run(self):

        """Calls :mod:`al.learning_curve.run_trials`, :mod:`utils.utils.assign_plot_params`, :mod:`utils.utils.data_to_py`, and :mod:`front_end.gui.run_al_gui.HelperFunctions.gray_out`"""
        clas_strat = []
        for clas in clas_params_clas:
          for strat in strat_params_strat:
            if clas_params_clas[clas].get() and strat_params_strat[strat].get():
              clas_name = re.findall('(\w+)CheckVal_run', clas)[0]
              strat_name = re.findall('(\w+)CheckVal_run', strat)[0]
              clas_strat.append((clas_name, strat_name))

        run_list = open('files/run_list.txt', 'r')
        run_list_r = run_list.read()
        run_list.close()

        for item in clas_strat:
            pf = "%s-%s" % (item[0], item[1])
            run_list = open('files/run_list.txt', 'a')
            self.plotfile_inputvar.set(pf)

            args = ('-pf', self.plotfile_inputvar.get(), '-c', item[0], '-d', self.menu_w.helper.train_load_val.get() + ' ' + self.menu_w.helper.test_load_val.get(), '-sd', self.menu_w.helper.single_load_val.get(), '-f', self.menu_w.pref_w.file_inputvar.get(), '-nt', int(run_params["nt_val"][0].get()), '-st', item[1], '-bs', int(run_params["bs_val"][0].get()), '-b', int(run_params["b_val"][0].get()), '-sz', int(run_params["sz_val"][0].get()), '-sp', int(run_params["sp_val"][0].get()))

            run_cmd = "python run_al_cl.py"

            for index, arg in enumerate(args):
              if index % 2 != 0 and arg != "''":
                run_cmd += ' %s %s' % (args[index-1], arg)

            if run_cmd not in run_list_r:
                print run_cmd

                learning_api = LearningCurve()

                classifier_name = eval((item[0]))
                alpha = {}

                values, avg_accu, avg_auc = learning_api.run_trials(self.menu_w.helper.X_pool, self.menu_w.helper.y_pool, self.menu_w.helper.X_test, self.menu_w.helper.y_test, item[1], classifier_name, alpha, int(run_params["bs_val"][0].get()), int(run_params["sz_val"][0].get()), int(run_params["b_val"][0].get()), int(run_params["nt_val"][0].get()))

                accu_x, accu_y, auc_x, auc_y = assign_plot_params(avg_accu, avg_auc)

                # Write data to plot_vals.py for plots
                plot_f = "plot_vals.py"
                data_to_py(plot_f, item[0], item[1], accu_x, accu_y, auc_x, auc_y)

                run_list.write(run_cmd + '\n')
                run_list.close()

        self.menu_w.helper.gray_out()


    def reset(self):
        """Resets the gui; calls :mod:`front_end.gui.run_al_gui.MainCanvas.clean`"""
        self.clean(run_params)
        self.clean(clas_params_clas)
        self.clean(strat_params_strat)
        self.clean(show_params_clas)
        self.clean(show_params_strat)

        self.menu_w.helper.train_load_val.set("''")
        self.menu_w.helper.test_load_val.set("''")
        self.menu_w.helper.single_load_val.set("''")

        run_list_f = open('files/run_list.txt', 'w')
        run_list_f.write('')
        run_list_f.close()

        plot_vals_f = open('plot_vals.py', 'w')
        plot_vals_f.write('vals = {}\n')
        plot_vals_f.close()

        self.menu_w.helper.gray_out()
        self.menu_w.helper.gray_run()

    def save_auc(self):
        """Saves auc plot; calls :mod:`front_end.gui.run_al_gui.MainCanvas.show_plots`"""
        auc_f = tkFileDialog.asksaveasfile(mode='w', defaultextension=".png")
        if auc_f != ():
          self.show_plots(auc_f)

    def save_acc(self):
        """Saves accuracy plot: calls :mod:`front_end.gui.run_al_gui.MainCanvas.show_plots`"""
        acc_f = tkFileDialog.asksaveasfile(mode='w', defaultextension=".png")
        if acc_f != ():
          self.show_plots(False, acc_f)

    def clean(self, params_dict):
        """Cleans parameter values

        **Parameters**

        params_dict (*dict*) - parameters to be reset

        """
        try:
          os.remove('img/plot_acc.png')
          os.remove('img/plot_auc.png')
        except:
          pass
        for param in params_dict:
          if type(params_dict[param]) is tuple:
            params_dict[param][0].set(params_dict[param][1])
          else:
            params_dict[param].set(0)

    def add_classifier_frame_2(self, master):
        """Create show_plots classifier frame

        **Parameters**

        * master - main Tkinter window

        """
        self.classifier_frame_2 = Frame(master)
        self.w.create_window(class_col, 100, anchor=NW, window=self.classifier_frame_2)

        show_params_clas["MultinomialNBCheckVal_2"] = IntVar()
        show_params["MultinomialNBCheckBox_2"] = Checkbutton(self.classifier_frame_2, variable=show_params_clas["MultinomialNBCheckVal_2"], text="MultinomialNB")
        show_params["MultinomialNBCheckBox_2"].pack(anchor=W)

        show_params_clas["KNeighborsClassifierCheckVal_2"] = IntVar()
        show_params["KNeighborsClassifierCheckBox_2"] = Checkbutton(self.classifier_frame_2, variable=show_params_clas["KNeighborsClassifierCheckVal_2"], text="KNeighborsClassifier")
        show_params["KNeighborsClassifierCheckBox_2"].pack(anchor=W)

        show_params_clas["LogisticRegressionCheckVal_2"] = IntVar()
        show_params["LogisticRegressionCheckBox_2"] = Checkbutton(self.classifier_frame_2, variable=show_params_clas["LogisticRegressionCheckVal_2"], text="LogisticRegression")
        show_params["LogisticRegressionCheckBox_2"].pack(anchor=W)

        show_params_clas["SVCCheckVal_2"] = IntVar()
        show_params["SVCCheckBox_2"] = Checkbutton(self.classifier_frame_2, variable=show_params_clas["SVCCheckVal_2"], text="SVC")
        show_params["SVCCheckBox_2"].pack(anchor=W)

        show_params_clas["BernoulliNBCheckVal_2"] = IntVar()
        show_params["BernoulliNBCheckBox_2"] = Checkbutton(self.classifier_frame_2, variable=show_params_clas["BernoulliNBCheckVal_2"], text="BernoulliNB")
        show_params["BernoulliNBCheckBox_2"].pack(anchor=W)

        show_params_clas["DecisionTreeClassifierCheckVal_2"] = IntVar()
        show_params["DecisionTreeClassifierCheckBox_2"] = Checkbutton(self.classifier_frame_2, variable=show_params_clas["DecisionTreeClassifierCheckVal_2"], text="DecisionTreeClassifier")
        show_params["DecisionTreeClassifierCheckBox_2"].pack(anchor=W)

        show_params_clas["RandomForestClassifierCheckVal_2"] = IntVar()
        show_params["RandomForestClassifierCheckBox_2"] = Checkbutton(self.classifier_frame_2, variable=show_params_clas["RandomForestClassifierCheckVal_2"], text="RandomForestClassifier")
        show_params["RandomForestClassifierCheckBox_2"].pack(anchor=W)

        show_params_clas["AdaBoostClassifierCheckVal_2"] = IntVar()
        show_params["AdaBoostClassifierCheckBox_2"] = Checkbutton(self.classifier_frame_2, variable=show_params_clas["AdaBoostClassifierCheckVal_2"], text="AdaBoostClassifier")
        show_params["AdaBoostClassifierCheckBox_2"].pack(anchor=W)

        show_params_clas["GaussianNBCheckVal_2"] = IntVar()
        show_params["GaussianNBCheckBox_2"] = Checkbutton(self.classifier_frame_2, variable=show_params_clas["GaussianNBCheckVal_2"], text="GaussianNB")
        show_params["GaussianNBCheckBox_2"].pack(anchor=W)

    def add_strategy_frame_2(self, master):
        """Create show_plots strategy frame

        **Parameters**

        * master - main Tkinter window

        """
        self.strategy_frame_2 = Frame(master)
        self.w.create_window(strat_col, 100, anchor=NW, window=self.strategy_frame_2)

        show_params_strat["randCheckVal_2"] = IntVar()
        show_params["randCheckBox_2"] = Checkbutton(self.strategy_frame_2, variable=show_params_strat["randCheckVal_2"], text="rand")
        show_params["randCheckBox_2"].pack(anchor=W)

        show_params_strat["erreductCheckVal_2"] = IntVar()
        show_params["erreductCheckBox_2"] = Checkbutton(self.strategy_frame_2, variable=show_params_strat["erreductCheckVal_2"], text="erreduct")
        show_params["erreductCheckBox_2"].pack(anchor=W)

        show_params_strat["loggainCheckVal_2"] = IntVar()
        show_params["loggainCheckBox_2"] = Checkbutton(self.strategy_frame_2, variable=show_params_strat["loggainCheckVal_2"], text="loggain")
        show_params["loggainCheckBox_2"].pack(anchor=W)

        show_params_strat["qbcCheckVal_2"] = IntVar()
        show_params["qbcCheckBox_2"] = Checkbutton(self.strategy_frame_2, variable=show_params_strat["qbcCheckVal_2"], text="qbc")
        show_params["qbcCheckBox_2"].pack(anchor=W)

        show_params_strat["uncCheckVal_2"] = IntVar()
        show_params["uncCheckBox_2"] = Checkbutton(self.strategy_frame_2, variable=show_params_strat["uncCheckVal_2"], text="unc")
        show_params["uncCheckBox_2"].pack(anchor=W)

    def add_run_classifier_frame(self, master):
        """Creates run classifier frame

        **Parameters**

        * master - main Tkinter window

        """
        self.run_classifier_frame = Frame(master)
        self.w.create_window(run_class_col, 350, anchor=NW, window=self.run_classifier_frame)

        clas_params_clas["MultinomialNBCheckVal_run"] = IntVar()
        clas_params_clas["MultinomialNBCheckVal_run"].set(1)
        clas_params["MultinomialNBCheckBox_run"] = Checkbutton(self.run_classifier_frame, variable=clas_params_clas["MultinomialNBCheckVal_run"], text="MultinomialNB")
        clas_params["MultinomialNBCheckBox_run"].pack(anchor=W)

        clas_params_clas["KNeighborsClassifierCheckVal_run"] = IntVar()
        clas_params["KNeighborsClassifierCheckBox_run"] = Checkbutton(self.run_classifier_frame, variable=clas_params_clas["KNeighborsClassifierCheckVal_run"], text="KNeighborsClassifier")
        clas_params["KNeighborsClassifierCheckBox_run"].pack(anchor=W)

        clas_params_clas["LogisticRegressionCheckVal_run"] = IntVar()
        clas_params["LogisticRegressionCheckBox_run"] = Checkbutton(self.run_classifier_frame, variable=clas_params_clas["LogisticRegressionCheckVal_run"], text="LogisticRegression")
        clas_params["LogisticRegressionCheckBox_run"].pack(anchor=W)

        clas_params_clas["SVCCheckVal_run"] = IntVar()
        clas_params["SVCCheckBox_run"] = Checkbutton(self.run_classifier_frame, variable=clas_params_clas["SVCCheckVal_run"], text="SVC")
        clas_params["SVCCheckBox_run"].pack(anchor=W)

        clas_params_clas["BernoulliNBCheckVal_run"] = IntVar()
        clas_params["BernoulliNBCheckBox_run"] = Checkbutton(self.run_classifier_frame, variable=clas_params_clas["BernoulliNBCheckVal_run"], text="BernoulliNB")
        clas_params["BernoulliNBCheckBox_run"].pack(anchor=W)

        clas_params_clas["DecisionTreeClassifierCheckVal_run"] = IntVar()
        clas_params["DecisionTreeClassifierCheckBox_run"] = Checkbutton(self.run_classifier_frame, variable=clas_params_clas["DecisionTreeClassifierCheckVal_run"], text="DecisionTreeClassifier")
        clas_params["DecisionTreeClassifierCheckBox_run"].pack(anchor=W)

        clas_params_clas["RandomForestClassifierCheckVal_run"] = IntVar()
        clas_params["RandomForestClassifierCheckBox_run"] = Checkbutton(self.run_classifier_frame, variable=clas_params_clas["RandomForestClassifierCheckVal_run"], text="RandomForestClassifier")
        clas_params["RandomForestClassifierCheckBox_run"].pack(anchor=W)

        clas_params_clas["AdaBoostClassifierCheckVal_run"] = IntVar()
        clas_params["AdaBoostClassifierCheckBox_run"] = Checkbutton(self.run_classifier_frame, variable=clas_params_clas["AdaBoostClassifierCheckVal_run"], text="AdaBoostClassifier")
        clas_params["AdaBoostClassifierCheckBox_run"].pack(anchor=W)

        clas_params_clas["GaussianNBCheckVal_run"] = IntVar()
        clas_params["GaussianNBCheckBox_run"] = Checkbutton(self.run_classifier_frame, variable=clas_params_clas["GaussianNBCheckVal_run"], text="GaussianNB")
        clas_params["GaussianNBCheckBox_run"].pack(anchor=W)

    def add_run_strategy_frame(self, master):
        """Creates run strategy frame

        **Parameters**

        * master - main Tkinter window

        """
        self.run_strategy_frame = Frame(master)
        self.w.create_window(run_strat_col, 350, anchor=NW, window=self.run_strategy_frame)

        strat_params_strat["randCheckVal_run"] = IntVar()
        strat_params_strat["randCheckVal_run"].set(1)
        strat_params["randCheckBox_run"] = Checkbutton(self.run_strategy_frame, variable=strat_params_strat["randCheckVal_run"], text="rand")
        strat_params["randCheckBox_run"].pack(anchor=W)

        strat_params_strat["erreductCheckVal_run"] = IntVar()
        strat_params["erreductCheckBox_run"] = Checkbutton(self.run_strategy_frame, variable=strat_params_strat["erreductCheckVal_run"],text="erreduct")
        strat_params["erreductCheckBox_run"].pack(anchor=W)

        strat_params_strat["loggainCheckVal_run"] = IntVar()
        strat_params["loggainCheckBox_run"] = Checkbutton(self.run_strategy_frame, variable=strat_params_strat["loggainCheckVal_run"], text="loggain")
        strat_params["loggainCheckBox_run"].pack(anchor=W)

        strat_params_strat["qbcCheckVal_run"] = IntVar()
        strat_params["qbcCheckBox_run"] = Checkbutton(self.run_strategy_frame, variable=strat_params_strat["qbcCheckVal_run"], text="qbc")
        strat_params["qbcCheckBox_run"].pack(anchor=W)

        strat_params_strat["uncCheckVal_run"] = IntVar()
        strat_params["uncCheckBox_run"] = Checkbutton(self.run_strategy_frame, variable=strat_params_strat["uncCheckVal_run"], text="unc")
        strat_params["uncCheckBox_run"].pack(anchor=W)

    def add_buttons(self):
        """Creates buttons; calls :mod:`front_end.gui.run_al_gui.MainCanvas.show_plots`, :mod:`front_end.gui.run_al_gui.MainCanvas.clear_plots`, :mod:`front_end.gui.run_al_gui.MainCanvas.run`, :mod:`front_end.gui.run_al_gui.MainCanvas.reset`, :mod:`front_end.gui.run_al_gui.MainCanvas.save_auc`, :mod:`front_end.gui.run_al_gui.MainCanvas.save_acc`"""
        self.showButton = Button(text="Show Plots", command=self.show_plots)
        self.w.create_window(show_col-50, 300, anchor=NW, window=self.showButton)

        self.clearButton = Button(text="Clear Plots", command=self.clear_plots)
        self.w.create_window(show_col+50, 300, anchor=NW, window=self.clearButton)

        self.runButton = Button(text="Run", command=self.run)
        self.w.create_window(run_col-20, 555, anchor=NW, window=self.runButton)

        self.resetButton = Button(text="Reset", command=self.reset)
        self.w.create_window(run_col+40, 555, anchor=NW, window=self.resetButton)

        self.saveAucButton = Button(text="Save Auc Plot", command=self.save_auc)
        self.w.create_window(show_col-67, 335, anchor=NW, window=self.saveAucButton)

        self.saveAccButton = Button(text="Save Acc Plot", command=self.save_acc)
        self.w.create_window(show_col+50, 335, anchor=NW, window=self.saveAccButton)

    def add_alerts(self):
        """Creates alerts(labels)"""
        self.single_load_label = Label(text="Loaded Single Dataset: ", font=self.times_i_10, bg="grey")
        self.w.create_window(20, 20, anchor = NW, window=self.single_load_label)

        self.menu_w.helper.single_load_val.set("''")
        self.single_load = Label(textvariable=self.menu_w.helper.single_load_val, width=label_limit, bg="#00FF33")
        self.w.create_window(20, 50, anchor=NW, window=self.single_load)

        self.train_load_label = Label(text="Loaded Train Dataset: ", font=self.times_i_10, bg="grey")
        self.w.create_window(20, 90, anchor=NW, window=self.train_load_label)

        self.menu_w.helper.train_load_val.set("''")
        self.train_load = Label(textvariable=self.menu_w.helper.train_load_val, width = label_limit, bg="#00FF33")
        self.w.create_window(20, 120, anchor=NW, window=self.train_load)

        self.test_load_label = Label(text="Loaded Test Dataset: ", font=self.times_i_10, bg="grey")
        self.w.create_window(20, 160, anchor=NW, window=self.test_load_label)

        self.menu_w.helper.test_load_val.set("''")
        self.test_load = Label(textvariable=self.menu_w.helper.test_load_val, width=label_limit, bg="#00FF33")
        self.w.create_window(20, 190, anchor=NW, window=self.test_load)

        self.plotfile_inputvar = StringVar()


class Main(object):
    """Integrates all objects together"""
    def __init__(self):
        """Instantiates :mod:`front_end.gui.run_al_gui.Main`"""
        self.master = Tk()
        self.master.title("Python GUI")
        self.master.geometry('1100x600+100+1')
        self.master.protocol('WM_DELETE_WINDOW', self.exit_master)
        #self.menu_w = MenuWindow(self.master)
        self.main_c = MainCanvas(self.master)
        #self.helper = HelperFunctions()

    def exit_master(self):
        """Closes gui"""
        self.master.destroy()
        try:
          pid = os.getpid()
          os.kill(pid, signal.SIGTERM)
        except:
          pass

    def show_menubar(self):
        """Configures menubar"""
        self.master.config(menu=self.main_c.menu_w.menubar)

    def run(self):
        """Calls :mod:`front_end.gui.run_al_gui.Main.show_menubar`"""
        self.show_menubar()

if __name__ == '__main__':
    gui = Main()
    gui.run()
    gui.main_c.menu_w.helper.gray_out()
    gui.master.mainloop()
