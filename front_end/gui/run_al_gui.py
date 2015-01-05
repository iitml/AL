"""
The GUI module to run the active learning strategies.
"""

from __init__ import *

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

class ParamsWindow(object):
    def __init__(self):
        '''Parameters Window'''
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
        self.pref.deiconify()

    def exit_pref(self):
        self.pref.withdraw()

    def check_int(self, param):
        try:
          int(param[0].get())
          return True
        except ValueError:
          tkMessageBox.showinfo("Error", "%s value is not a number!\nSetting Default" % param[2])
          param[0].set(param[1])
          return False

    def display_params(self):
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


class MenuWindow(object):
    def __init__(self, master):
        self.menubar = Menu(master)
        self.pref_w = ParamsWindow()

        self.show_filemenu(master)
        self.show_editmenu(master)

    def show_filemenu(self, master):
        self.filemenu = Menu(self.menubar, tearoff=0)
        self.filemenu.add_command(label="Load single dataset", command=lambda: open_data('single'))
        self.filemenu.add_command(label="Load train dataset", command=lambda: open_data('train'))
        self.filemenu.add_command(label="Load test dataset", command=lambda: open_data('test'))
        self.filemenu.add_command(label="Quit           (ESC)", command=master.quit)
        self.menubar.add_cascade(label="File", menu=self.filemenu)

    def show_editmenu(self, master):
        self.editmenu = Menu(self.menubar, tearoff=0)
        self.editmenu.add_command(label="Parameters", command=self.pref_w.display_pref)
        self.menubar.add_cascade(label="Edit", menu=self.editmenu)

class MainCanvas(object):
    def __init__(self, master):
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

    def show_plots(self, auc_save=False, acc_save=False):
        pass

    def clear_plots(self):
        pass

    def run(self):
        pass

    def reset(self):
        self.clean(run_params)
        self.clean(clas_params_clas)
        self.clean(strat_params_strat)
        self.clean(show_params_clas)
        self.clean(show_params_strat)

        self.train_load_val.set("''")
        self.test_load_val.set("''")
        self.single_load_val.set("''")

        run_list_f = open('files/run_list.txt', 'w')
        run_list_f.write('')
        run_list_f.close()

        plot_vals_f = open('files/plot_vals.py', 'w')
        plot_vals_f.write('vals = {}\n')
        plot_vals_f.close()

        self.gray_out()
        self.gray_run()

    def save_auc(self):
        pass

    def save_acc(self):
        pass

    def clean(self, params_dict):
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

    def all_combos(self):
        result = []
        for clas in clas_params_clas:
          for strat in strat_params_strat:
            clas_name = re.findall('(\w+)CheckVal_run', clas)[0]
            strat_name = re.findall('(\w+)CheckVal_run', strat)[0]
            result.append((clas_name, strat_name))
        return result

    def gray_out(self):
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

    def gray_run(self):
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

    def add_classifier_frame_2(self, master):
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
        self.single_load_label = Label(text="Loaded Single Dataset: ", font=self.times_i_10, bg="grey")
        self.w.create_window(20, 20, anchor = NW, window=self.single_load_label)

        self.single_load_val = StringVar()
        self.single_load_val.set("''")
        self.single_load = Label(textvariable=self.single_load_val, width=label_limit, bg="#00FF33")
        self.w.create_window(20, 50, anchor=NW, window=self.single_load)

        self.train_load_label = Label(text="Loaded Train Dataset: ", font=self.times_i_10, bg="grey")
        self.w.create_window(20, 90, anchor=NW, window=self.train_load_label)

        self.train_load_val = StringVar()
        self.train_load_val.set("''")
        self.train_load = Label(textvariable=self.train_load_val, width = label_limit, bg="#00FF33")
        self.w.create_window(20, 120, anchor=NW, window=self.train_load)

        self.test_load_label = Label(text="Loaded Test Dataset: ", font=self.times_i_10, bg="grey")
        self.w.create_window(20, 160, anchor=NW, window=self.test_load_label)

        self.test_load_val = StringVar()
        self.test_load_val.set("''")
        self.test_load = Label(textvariable=self.test_load_val, width=label_limit, bg="#00FF33")
        self.w.create_window(20, 190, anchor=NW, window=self.test_load)

        self.plotfile_inputvar = StringVar()


class Main(object):
    def __init__(self):
        self.master = Tk()
        self.master.title("Python GUI")
        self.master.geometry('1100x600+100+1')
        self.master.protocol('WM_DELETE_WINDOW', self.exit_master)
        self.menu_w = MenuWindow(self.master)
        self.main_c = MainCanvas(self.master)

    def exit_master(self):
        self.master.destroy()
        try:
          pid = os.getpid()
          os.kill(pid, signal.SIGTERM)
        except:
          pass

    def show_menubar(self):
        self.master.config(menu=self.menu_w.menubar)

    def run(self):
        self.show_menubar()

if __name__ == '__main__':
    gui = Main()
    gui.run()
    gui.main_c.gray_out()
    gui.master.mainloop()
