# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:01:21 2018

@author: Eve Fleisig

This module allows the user to examine and visualize similarity or 
dissimilarity scores for data records with different file types.
"""

import gc
import numpy as np
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import math
import inspect
from textwrap import wrap
from functools import partial

import PyQt5.QtGui as qtgui
import PyQt5.QtWidgets as qtw
import PyQt5.QtCore as qtcore
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as 
    FigureCanvas)
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as 
    NavigationToolbar)

from matplotlib import colors as mcolors
from matplotlib import path
from matplotlib.widgets import (LassoSelector, RectangleSelector, Button, 
    MultiCursor)
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from sklearn.manifold.mds import MDS
from sklearn.cluster import KMeans, spectral_clustering, dbscan
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import pairwise_distances
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree

import vemos.GUIBackend as guibackend
import vemos.DataRecordBrowser as drb

class VisualMetricAnalyzer(qtw.QMainWindow):
    """The main GUI for examining a data set's similarity/dissimilarity scores.

    Parameters
    ----------
    data_set : DataSet
        The set of data (records, groupings, matrices, and data types) 
        to use in the GUI.

    Attributes
    ----------
    data_set : DataSet
        The set of data (records, groupings, matrices, and data types) 
        to use in the GUI.
    show_table : bool
        True if the GUI should display a table of scores and False if it
        should display a list.
    plots_per_fig : int
        The number of plots to display per matplotlib figure.
    selection_id : int
        The ID of the current figure being displayed.
        
    cluster_method : str
        The current method to use for clustering.
    cluster_methods : dict of {str : function}
        The clustering methods available.
    spectral_prefs : dict of {str : list}
        The current preferences for spectral clustering.
    hier_prefs : dict of {str : list}
        The current preferences for hierarchical clustering.
    settings : dict of {str : list}
        The current plot display settings.
    color_options : dict of {str : list}
        The options for coloring different plot elements.
    marker_style_options : dict of {str : str}
        The style options for different plot elements.
    color_grouping : str
        The name of the grouping to use when coloring a visualization by
        group or reordering the heat map by group.
        
    selected_records : list of list or int
        Stores the currently selected set of records (as pairs or 
        record IDs or as single record IDs).
    selected_data_types : list of str
        Stores the currently selected data types to display.
    selected_matrices : list of str
        Stores the currently selected matrices to use for analyses.
        
    figure_info : dict
        Stores information about the matplotlib figures being displayed.
    selectors : list of matplotlib.widgets._SelectorWidget
        Maintains a reference to the current selector for each open 
        figure.
    artists : dict of {str : list}
        The current artists for the 2D, 3D, match visualization, and clustering 
        plots.
    annotations : dict of {str : list}
        The current annotations for the 2D, 3D, match visualization, and 
        clustering plots.
    gb : GUIBackend
        Instance of GUIBackend for display methods.
    """
    
    def __init__(self, data_set):

        super(VisualMetricAnalyzer, self).__init__()
        plt.ion()
        self.data_set = data_set
        self.gb = guibackend.GUIBackend()
        
        # Tracks information that should stay the same even after update_data
        self.show_table = True
        self.selection_id = 0
        
        self.cluster_method = "Spectral"
        self.cluster_methods = {"Hierarchical": self.hierarchical_clustering,
                                "Spectral": self.show_spectral_clustering}
        # eigen_solver must be arpack; assign_labels must be default
        self.spectral_prefs = {"n_clusters": [3], "n_components": [3], 
                               "n_init": [10], "eigen_tol": [0.0]}
        self.hier_prefs = {"method": ["Ward"], "p": [20], "truncate_mode": 
                           ["none"], "orientation": ["top"], "no_labels": 
                           [True], "leaf_rotation": [90.]}
        self.settings = {"Plots per page": [4], "Marker color": ["Red"], 
                        "Marker shape": ["Point"], "Heat map coloring": 
                        ["Color"]}
            
        self.color_options = {"Blue": "b", "Red": "r", "Green": "g", 
                              "Cyan": "c", "Magenta": "m", "Yellow": "y",
                              "Black": "k", "White": "w"}                
        self.marker_style_options = {
            "Point": ".", "Pixel": ",", "Circle": "o", "Triangle": "^", 
            "Octagon": "8", "Square": "s", "Pentagon": "p", "Star": "*", 
            "Hexagon": "h", "+": "+", "x": "x", "Diamond": "D", "|": "|", 
            "_": "_"}
        
        # Creates menu bar
        menu_bar = self.menuBar() # change to qtgui.QMenuBar() for styling
        
        # File menu: To choose a score matrix or data description file
        file_menu = menu_bar.addMenu("File")
        file_menu.addAction(self.gb.make_widget(
            qtw.QAction('Update loaded data', self),
            "triggered", self.update_data))
        file_menu.addAction(self.gb.make_widget(
            qtw.QAction('Save analyses', self),
            "triggered", partial(self.data_set.save, self, False)))
        
        menu_bar.addAction(self.gb.make_widget(
            qtw.QAction('Display Settings', self), "triggered", 
            self.change_settings))
        
        # Opens an instance of the Data Record Browser                                             
        self.browser_action = self.gb.make_widget(
            qtw.QAction('Record Browser', self), "triggered", self.open_vrb)
        menu_bar.addAction(self.browser_action)
               
        # Clustering: Opens a new widget for displaying clusters
        menu_bar.addAction(self.gb.make_widget(qtw.QAction(
            'Clustering', self), "triggered", self.cluster_options))
        
        # Matrix creation: Generates or fuses matrices
        create_menu = menu_bar.addMenu("Edit Matrices")
        self.gen_action = self.gb.make_widget(qtw.QAction(
            'Generate Matrix', self), "triggered", self.make_matrices)
        create_menu.addAction(self.gen_action)
        self.fuse_action = self.gb.make_widget(qtw.QAction(
            'Fuse Matrices', self), "triggered", self.fuse_matrices)
        create_menu.addAction(self.fuse_action)
        
        
        # Analyses: To view 2D and 3D MDS, the heat map, or the histogram
        analysis_menu = menu_bar.addMenu("Analyses")
        mds_menu = analysis_menu.addMenu("Multidimensional Scaling")

        mds_menu.addAction(self.gb.make_widget(
            qtw.QAction('View 2D MDS', self), "triggered", self.show_2D_MDS))
        mds_menu.addAction(self.gb.make_widget(
            qtw.QAction('View 3D MDS', self), "triggered", self.show_3D_MDS))
        analysis_menu.addAction(self.gb.make_widget(
            qtw.QAction('Heat Map', self), "triggered", self.heat_map))
        
        # Match Visualizations: ROC curve, histograms, linear ordering, stats
        self.match_menu = menu_bar.addMenu("Binary Classification Performance")
        
        match_actions = {"ROC Curve": 
                         self.show_roc_curve, "Linear Ordering": 
                         self.linear_ordering, "Histogram": 
                         self.histogram, "Smooth Histogram": 
                         self.smooth_histogram, "Statistics": self.show_stats}
                        
        for action in match_actions:
            self.match_menu.addAction(self.gb.make_widget(qtw.QAction(
                action, self), "triggered", match_actions[action]))
        
        desktop = qtw.QDesktopWidget()
        self.screen_geometry = desktop.screenGeometry()
        self.setGeometry(40, 70, 850, 650) # old 900 700
        self.setWindowTitle("Visual Metric Analyzer")
        self.setWindowIcon(self.data_set.icon)
        self.create_main_frame()
        self.show()
        self.data_set.num_widgets_open += 1
        
###############################################################################
# Loading and table option methods
        
    def create_main_frame(self):
        """ Creates the main frame of the GUI.
        
        Creates all tabs, tables, and buttons. Unlike __init__, this method 
        contains all information that might change after calling update_data.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        See Also
        --------
        update_data: updates the data loaded in the widget.
        
        """
        
        self.selected_data_types = []
        self.selected_matrices = []
        self.selected_records = []
        self.figure_info = {} 
        self.text_widgets = []
        
        self.current_axes = None # For mv_widget
        self.current_2d_axes = None
        self.current_3d_axes = None
        self.current_cluster_axes = None
        
        self.title = 'Visual Metric Analyzer: ' + self.data_set.name
        self.setWindowTitle(self.title)
        
        self.color_grouping = None
        
        self.mv_widget = None
        self.interpolation_scores = []
        self.cur_points = []
        self.use_1_plot = False
        self.cluster_grouping_index = 1
        
        self.fig_2d = None
        self.fig_3d = None
        
        self.artists = {"2d": [], "3d": [], "mv": [], "cluster": []}
        self.annotations = {"2d": [], "3d": [], "mv": [], "cluster": []}
        self.pt_index_mv = None
        self.pt_index_cluster = None
        self.pt_index_2d = None
        self.pt_index_3d = None
        self.cur_note = None
        self.cur_mv_note = None
        
        self.selectors = []
        
        self.cluster_widget = None
        self.options_widget = None
        self.display_title = "View Records"
        
        # Checks if ground truth values are available for match visualizations
        for matrix in self.data_set.matrices:
            if ((not 0 in self.data_set.match_nonmatch_scores[matrix]["gt"]) or 
                (not 1 in self.data_set.match_nonmatch_scores[matrix]["gt"])):
                self.match_menu.setDisabled(True)
                break
        else:
            self.match_menu.setDisabled(False)
            
        if self.data_set.has_files==False:
            self.browser_action.setDisabled(True)
        else:
            self.browser_action.setDisabled(False)
            
        # Checks if records/matrices available for matrix generation/fusion
        if not self.data_set.has_files:
            self.gen_action.setDisabled(True)
        else:
            self.gen_action.setDisabled(False)
            
        if len(self.data_set.matrices) < 2:
            self.fuse_action.setDisabled(True)
        else:
            self.fuse_action.setDisabled(False)
        
        # Makes central table, check boxes, and buttons
        centerWidget = qtw.QWidget()

        layout = qtw.QHBoxLayout()
        layout.setContentsMargins(30, 20, 30, 30)
        
        font = qtgui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        
        # Makes tabs with tables for each score matrix
        self.matrix_tabs = qtw.QTabWidget()
        
        for matrix_name in self.data_set.matrices:
            table = qtw.QTableWidget(self)
            table.setGeometry(qtcore.QRect(50, 70, 600, 591))
            
            if self.data_set.has_files:
                table.itemSelectionChanged.connect(self.table_selection)
                table.itemPressed.connect(self.table_pressed)

            table.setEditTriggers(qtw.QAbstractItemView.NoEditTriggers)
    
            self.matrix_tabs.addTab(table, matrix_name)
        
        opt_scroll = qtw.QScrollArea(self) 
        opt_scroll.setWidgetResizable(True)        
        opt_scroll_container = qtw.QWidget()
        opt_scroll.setWidget(opt_scroll_container)
        opt_layout = qtw.QVBoxLayout(opt_scroll_container)
        
        opt_layout.addWidget(
            self.gb.make_widget(qtw.QLabel("Options"), font=font))
        
        # Adds check boxes to choose which file types to view
        font.setPointSize(10)
        opt_layout.addWidget(self.gb.make_widget(qtw.QLabel(
                             "Data Types to Display:"), font=font))
        for name in self.data_set.data_types:
            new_box = self.gb.make_widget(
                qtw.QCheckBox(self), "stateChanged", 
                self.data_type_boxes_checked, "Show " + name)  
            opt_layout.addWidget(new_box)
            new_box.setChecked(True)
            
        opt_layout.addSpacing(20) 

        # Adds check boxes to choose which matrices to use for analyses 
        opt_layout.addWidget(self.gb.make_widget(qtw.QLabel(
                             "Matrices to Use in Analyses:"), font=font))
        for name in self.data_set.matrices:
            new_box = self.gb.make_widget(
                qtw.QCheckBox(self), "stateChanged", 
                self.matrix_boxes_checked, name)
            new_box.setChecked(True)
            #self.selected_matrices.append(name)
            opt_layout.addWidget(new_box)
            
        opt_layout.addSpacing(20)
    
        # Adds buttons to view scores as table or list  
        opt_layout.addWidget(self.gb.make_widget(qtw.QLabel(
            "Display Format:"), font=font))
        
        
        if "Matrix" in [self.data_set.matrices[matrix]["format"] 
                        for matrix in self.data_set.matrices]:
            self.table_button = self.gb.make_widget(
                qtw.QRadioButton(self), "toggled", self.table_list_change, 
                "Table", checked=True)
            self.list_button = self.gb.make_widget(
                qtw.QRadioButton(self), "toggled", self.table_list_change, 
                "List (existing values only)")
        else:
            self.table_button = self.gb.make_widget(
                qtw.QRadioButton(self), "toggled", self.table_list_change, 
                "Table")
            self.list_button = self.gb.make_widget(
                qtw.QRadioButton(self), "toggled", self.table_list_change, 
                "List (existing values only)", checked=True)
            
        opt_layout.addWidget(self.table_button)
        opt_layout.addWidget(self.list_button)
        opt_layout.addStretch(2)
        
        layout.addWidget(opt_scroll, 1)
        layout.addWidget(self.matrix_tabs, 3)

        centerWidget.setLayout(layout)
        self.setCentralWidget(centerWidget)
        
         
    def table_list_change(self, button):
        """Switches between table of scores and list of available scores only.
        
        Parameters
        ----------
        button : QButton
            The button clicked ("Table" or "List").
        
        Returns
        -------
        None
        """
        
        if button.text() == "Table":
            self.make_table()
        else:
            self.make_list()
            
        
    def make_table(self):
        """Displays a table of dissimilarity scores for each score matrix.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
        for index in range(len(self.matrix_tabs)):
            
            table = self.matrix_tabs.widget(index)
            matrix_name = str(self.matrix_tabs.tabText(index))
            cur_matrix = self.data_set.matrices[matrix_name]["matrix"]
            
            table.clear()
            table.setRowCount(0)
            table.setColumnCount(0)
            table.setRowCount(len(cur_matrix))
            table.setColumnCount(len(cur_matrix[0]))
            
            for row, line in enumerate(cur_matrix):
                for col, item in enumerate(line):
                    
                    if item is None or np.isnan(item):
                        widget_item = qtw.QTableWidgetItem("N/A")
                    else:
                        widget_item = qtw.QTableWidgetItem(str(item).strip())
                    table.setItem(row, col, widget_item)
            
            all_ids = [record.id for record in self.data_set.records]
            table.setHorizontalHeaderLabels(all_ids)
            table.setVerticalHeaderLabels(all_ids)
            table.setSortingEnabled(True)
    
    
    def make_list(self):
        """Displays list of available (dis)similarity scores for each matrix.
        
        Unlike the table, does not display pairs for which there is no score.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
            
        for index in range(len(self.matrix_tabs)):
            
            table = self.matrix_tabs.widget(index)
            matrix_name = str(self.matrix_tabs.tabText(index))
            cur_matrix = self.data_set.matrices[matrix_name]["matrix"]
        
            # Reshapes the GUI table
            table.clear()
            table.setRowCount(0)
            table.setColumnCount(2)
            table.setHorizontalHeaderLabels(["IDs", "Score"])
            header = table.horizontalHeader()
            header.setSectionResizeMode(0, qtw.QHeaderView.ResizeToContents)
            header.setSectionResizeMode(1, qtw.QHeaderView.Stretch)
            
            for row, line in enumerate(cur_matrix):
                for col, item in enumerate(line):
                    
                    # Ignores unavailable matches and matches with self
                    if item is not None and not np.isnan(item) and row != col:
                        
                        pair_text = ("  " + self.data_set.records[row].id + 
                            " and " + self.data_set.records[col].id + "  ")
                        item_text = "  " + str(item).strip() + "  "
                        
                        nrows = table.rowCount()
                        table.insertRow(nrows)
                        table.setItem(nrows,0,qtw.QTableWidgetItem(pair_text))
                        table.setItem(nrows,1,qtw.QTableWidgetItem(item_text))
                            
                                   
    def data_type_boxes_checked(self, box):
        """Updates the data types to display based on the boxes checked. 
        
        Parameters
        ----------
        box : QCheckBox
            The check box that was selected.
        
        Returns
        -------
        None
        """
        
        for data_type in self.data_set.data_types:
            if "show " + data_type.lower() == str(box.text()).lower():
                if box.isChecked():
                    self.selected_data_types.append(data_type)
                else:
                    self.selected_data_types.remove(data_type)
    
    
    def matrix_boxes_checked(self, box):
        """Updates the matrices to use for analysis based on the boxes checked.
        
        Parameters
        ----------
        box : QCheckBox
            The check box that was selected.
        
        Returns
        -------
        None
        """
        if box.isChecked():
            self.selected_matrices.append(str(box.text()))
        else:
            self.selected_matrices.remove(str(box.text()))
                    

###############################################################################
# File menu methods                                                             

    def update_data(self):
        """Allows the user to update the data currently loaded. 
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        See Also
        --------
        DataSet.create_data_loading_widget: Opens the widget for updating data.
        """
        
        plt.close("all")
        self.data_set.update = True
        self.data_set.create_data_loading_widget()
        
        if self.data_set.update:
            self.close()
            self.data_set.update = False
            
            
            # Sets up display
            self.selected_records=[]
            if self.data_set.interface_to_open == "Visual Metric Analyzer":
                self.create_main_frame()
                self.data_set.num_widgets_open += 1
                self.show()
            else:
                drb.DataRecordBrowser(self.data_set)
            
    
    def open_vrb(self):
        """ Opens the Data Record Browser.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
        plt.close("all")
        
        # Prompts user to load record files if needed
        if self.data_set.records == []:
            self.gb.general_msgbox("Add Record Files", ("Please load record "
                "files to use the Data Record Browser."))
            self.update_data()
        else:
            self.data_set.update = True
            drb.DataRecordBrowser(self.data_set)
            self.data_set.update = False

###############################################################################
    def change_settings(self):
        """Creates the widget for changing display settings.
        
        The user may change the plots displayed per page, the color and shape 
        of  matplotlib markers, and how heat maps are colored.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """         
        
        # Creates settings widget
        self.settings_widget = qtw.QWidget()
        self.settings_widget.setWindowTitle("Display")
        self.settings_widget.setWindowIcon(self.data_set.icon)
        settings_layout = qtw.QGridLayout()
        self.settings_widget.setLayout(settings_layout)

        font = qtgui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        
        title_label = self.gb.make_widget(
            qtw.QLabel("Display Settings"), font=font)
        title_label.setFixedHeight(50)
        settings_layout.addWidget(title_label, 0, 0, 1, 2)
        settings_layout.setSpacing(10)
        
        # Adds options
        settings_layout.addWidget(qtw.QLabel("Plots per page"), 1, 0)
        settings_layout.addWidget(self.gb.make_widget(
            qtw.QSpinBox(), "valueChanged", self.update_pref, 
            value=self.settings["Plots per page"][0], minimum=1, maximum=16,
            add_to=self.settings["Plots per page"], is_partial=True), 1, 1)
        
        settings_layout.addWidget(self.gb.make_widget(
            qtw.QLabel("Plotting Styles"), font=font), 2, 0)
        
        cur_color = str(self.settings["Marker color"][0]).capitalize()
        colors = list(self.color_options.keys())     
        settings_layout.addWidget(qtw.QLabel("Marker color"), 3, 0)
        settings_layout.addWidget(self.gb.make_widget(
                qtw.QComboBox(), "currentIndexChanged", self.update_pref, 
                items=colors, index=colors.index(cur_color), 
                add_to=self.settings["Marker color"], is_partial=True), 3, 1)
        
        cur_shape = str(self.settings["Marker shape"][0]).capitalize()
        shapes = list(self.marker_style_options.keys())        
        settings_layout.addWidget(qtw.QLabel("Marker shape"), 4, 0)
        settings_layout.addWidget(self.gb.make_widget(
                qtw.QComboBox(), "currentIndexChanged", self.update_pref, 
                items=shapes, index=shapes.index(cur_shape),
                add_to=self.settings["Marker shape"], is_partial=True), 4, 1)
        
        cur_heat = str(self.settings["Heat map coloring"][0]).capitalize()
        cur_heat_index = ["Color", "Grayscale"].index(cur_heat) 
        settings_layout.addWidget(qtw.QLabel("Heat map coloring"), 7, 0)
        settings_layout.addWidget(self.gb.make_widget(
            qtw.QComboBox(), "currentIndexChanged", self.update_pref, 
            items=["Grayscale", "Color"], index=cur_heat_index, 
            add_to=self.settings["Heat map coloring"], is_partial=True), 7, 1)
        
        self.settings_widget.show()


###############################################################################
# Analysis methods
   
    def heat_map(self):
        """ Displays the heat map of the current dissimilarity matrix.
        
        The color on the heat map indicates the score. If self.color_grouping 
        is true, reorders the heat map by grouping.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
        if len(self.selected_matrices) == 0:
            self.gb.no_selection_error()
            return
            
        fig = plt.figure(figsize=(8*len(self.selected_matrices), 8))
        fig.suptitle("Heat Map", fontsize=18)
        
        if self.settings["Heat map coloring"][0].capitalize() == "Grayscale":
            plt.gray()
        else:
            plt.set_cmap("jet")
        
        for index, matrix_name in enumerate(self.selected_matrices):
            
            cur_axis = fig.add_subplot(1, len(self.selected_matrices), index+1)
            cur_axis.set_xlabel(matrix_name)
            
            cur_matrix = self.data_set.matrices[matrix_name]["matrix"]
            
            
            if self.color_grouping:
                
                grouping_name = self.color_grouping
                
                if "Cluster" in self.color_grouping:
                    
                    try:
                        split_name = grouping_name.split(" ")
                        grouping_name = split_name[0] + " " + matrix_name 
                        grouping_name += " " + " ".join(split_name[-2:])
                    except ValueError:
                        self.gb.general_msgbox("Error reordering heat map", 
                                               "Please try again.")
                    
                labels = [0 for x in range(len(self.data_set.records))]
                grouping = self.data_set.groupings[grouping_name]
                for index, group in enumerate(grouping):
                    for cur_id in grouping[group]:
                        cur_index = self.data_set.index_from_id(cur_id)
                        labels[cur_index] == index

                D = cur_matrix

                # reordering_indices are clusters where IDs belong 
                # (speccluster labels)
                reordering_indices = np.argsort(labels)
                D_temp = D[ reordering_indices ]
                D_reordered = D_temp[ :,reordering_indices ]
                plt.imshow(D_reordered)
            
            else:
                plt.imshow(cur_matrix)
            

            plt.colorbar()
        
        fig.canvas.mpl_connect('button_press_event', self.heat_click)
        
#        color_axes = [.68, .01, .31, .051]
#        self.color_button = Button(plt.axes(color_axes),"Reorder by Grouping")        
#        self.color_button.on_clicked(self.color_by_grouping)

        plt.show()

        
    def show_2D_MDS(self):
        """Displays 2-dimensional scaling of the scores.
        
        Scales the distances between the data records to 2 dimensions. If 
        self.color_grouping is true, colors points according to what group they 
        are in (within that grouping). Activates the lasso or rectangle 
        selectors and the point selector.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
        if len(self.selected_matrices) == 0:
            self.gb.no_selection_error()
            return
        
        if self.fig_2d:
            plt.close(self.fig_2d)
            self.fig_2d = None
            gc.collect()
            
        # Creates the plot
        fsize = (8*len(self.selected_matrices), 8)
        self.fig_2d, self.current_2d_axes = plt.subplots(
            1, len(self.selected_matrices), figsize=fsize)
        if len(self.selected_matrices) == 1:
            self.current_2d_axes = [self.current_2d_axes]
        self.fig_2d.suptitle("2D Multidimensional Scaling", fontsize=18)
        
        self.annotations["2d"] = []
        self.artists["2d"] = []
        self.cur_points = []
        self.selectors = []
        cur_marker = self.marker_style_options[
            self.settings["Marker shape"][0].capitalize()]
            
        for matrix_index, matrix_name in enumerate(self.selected_matrices):

            cur_matrix = self.data_set.matrices[matrix_name]["matrix"]
            score_matrix0 = np.nan_to_num(np.array(cur_matrix))
            score_matrix = np.maximum(score_matrix0, score_matrix0.T)	
            mds2 = MDS(2, dissimilarity='precomputed')
            x, y = mds2.fit_transform(score_matrix).T # For 0-179
            self.cur_points.append([x, y])
            
            self.current_2d_axes[matrix_index].set_title(matrix_name)
            
            if self.color_grouping:
                
                # Colors by grouping, if needed
                grouping_name = self.color_grouping
                if "Cluster" in self.color_grouping:
                    split_name = grouping_name.split(" ")
                    grouping_name = split_name[0] + " " + matrix_name
                    grouping_name += " " + " ".join(split_name[-2:])
                    
                    if grouping_name not in self.data_set.groupings:
                        self.gb.general_msgbox("Scores under " + matrix_name + 
                                               "were not grouped under this "
                                               "grouping.")
                        continue

                # Adds IDs as annotations
                self.annotations["2d"].append({})
                self.artists["2d"].append({})
                grouping = self.data_set.groupings[grouping_name]
                for group in grouping:

                    x_coords = []
                    y_coords = []
                    
                    group_annotations = []
                    for cur_id in grouping[group]:
                        cur_index = self.data_set.index_from_id(cur_id)
                        x_coords.append(x[cur_index])
                        y_coords.append(y[cur_index])
                        
                        note = self.current_2d_axes[matrix_index].annotate(
                            cur_id, xy=(x[cur_index], y[cur_index]), 
                            arrowprops=dict(arrowstyle='->'),
                            bbox=dict(boxstyle="round", fc="w"))
                        note.set_visible(False)
                        group_annotations.append(note)
                    
                    self.annotations[
                        "2d"][matrix_index][group] = group_annotations
                    cur_artist, = self.current_2d_axes[matrix_index].plot(
                        x_coords, y_coords, marker=cur_marker, ls="", 
                        picker=10)
                    self.artists["2d"][matrix_index][group] = cur_artist
                    

            else:
                # Adds IDs as annotations
                cur_artist, = self.current_2d_axes[matrix_index].plot(x, y, 
                    marker=cur_marker, ls="", picker=10)
                self.artists["2d"].append(cur_artist)

                self.annotations["2d"].append([])
                for pt_index in range(len(x)):

                    cur_id = self.data_set.records[pt_index].id
                    note = self.current_2d_axes[matrix_index].annotate(cur_id, 
                        xy=(x[pt_index], y[pt_index]), arrowprops=dict(
                        arrowstyle='->'), bbox=dict(boxstyle="round", fc="w"))
                    note.set_visible(False)
                    self.annotations["2d"][matrix_index].append(note)

            self.selectors.append(RectangleSelector(
                self.current_2d_axes[matrix_index], 
                partial(self.plot_rect, matrix_index)))

        self.current_2d_axes[matrix_index].get_figure().canvas.draw()
        
        # Adds buttons and selectors
        axbutton = plt.axes([.88, .01, .11, .051])
        self.lasso_rect = Button(axbutton, "Lasso")
        self.lasso_rect.on_clicked(self.mds_selector_change)

        self.color_button = Button(plt.axes([.01, .01, .31, .051]), 
                                   "Color by Grouping")        
        self.color_button.on_clicked(partial(self.color_by_grouping, "2d"))
        
        self.fig_2d.canvas.mpl_connect("motion_notify_event", 
                                       partial(self.plot_hover, "2d"))
        self.fig_2d.canvas.mpl_connect('pick_event', self.plot_pick)

        plt.show()
 
              
    def show_3D_MDS(self):
        """Displays 3-dimensional scaling of the scores.
        
        Scales the distances between the data records to 2 dimensions. If 
        self.color_grouping is true, colors points according to what group they 
        are in (within that grouping). Activates the point selector.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
        if len(self.selected_matrices) == 0:
            self.gb.no_selection_error()
            return
        
        if self.fig_3d:
            plt.close(self.fig_3d)
            self.fig_3d = None
            gc.collect()
        
        # Creates the plot
        self.fig_3d = plt.figure(figsize=(8*len(self.selected_matrices), 8))
        self.fig_3d.suptitle("3D Multidimensional Scaling", fontsize=18)
        
        self.current_3d_axes = []
        self.annotations["3d"] = []
        self.artists["3d"] = []
        cur_marker = self.marker_style_options[
            self.settings["Marker shape"][0].capitalize()]
        for index, matrix_name in enumerate(self.selected_matrices):
            
            
            cur_matrix = self.data_set.matrices[matrix_name]["matrix"]
            self.current_3d_axes.append(self.fig_3d.add_subplot(1, 
                len(self.selected_matrices), index+1, projection='3d'))
    
            score_matrix0 = np.nan_to_num(np.array(cur_matrix))
            score_matrix = np.maximum(score_matrix0, score_matrix0.T)
    
            mds3 = MDS(3, dissimilarity='precomputed')
            x,y,z = mds3.fit_transform(score_matrix).T
            self.current_3d_axes[index].set_title(matrix_name)
            
            if self.color_grouping:
                grouping_name = self.color_grouping
                
                if "Cluster" in self.color_grouping:
                    split_name = grouping_name.split(" ")
                    grouping_name = split_name[0] + " " + matrix_name
                    grouping_name += " " + " ".join(split_name[-2:])
                
                # Adds IDs as annotations
                self.annotations["3d"].append({})
                self.artists["3d"].append({})
                grouping = self.data_set.groupings[grouping_name]
                for group in grouping:

                    x_coords = []
                    y_coords = []
                    z_coords = []
                    group_annotations = []
                    for cur_id in grouping[group]:
                        cur_index = self.data_set.index_from_id(cur_id)
                        x_coords.append(x[cur_index])
                        y_coords.append(y[cur_index])
                        z_coords.append(z[cur_index])
                        
                        note = self.current_3d_axes[index].text(x[cur_index], 
                            y[cur_index], z[cur_index], cur_id)
                        note.set_visible(False)
                        group_annotations.append(note)
                    
                    self.annotations[
                        "3d"][index][group] = group_annotations
                    cur_artist, = plt.plot(x_coords, y_coords, z_coords, 
                                           marker=cur_marker, ls="", picker=10)
                    self.artists["3d"][index][group] = cur_artist
            
            else:
                # Adds IDs as annotations
                cur_artist, = plt.plot(x, y, z, marker=cur_marker, ls="", 
                                       picker=10)
                self.artists["3d"].append(cur_artist)
                
                self.annotations["3d"].append([])
                for pt_index in range(len(x)):

                    cur_id = self.data_set.records[pt_index].id
                    note = self.current_3d_axes[index].text(
                        x[pt_index], y[pt_index], z[pt_index], cur_id)
                    note.set_visible(False)
                    self.annotations["3d"][index].append(note)
        
        color_axes = [.68, .01, .31, .051]
        self.color_button = Button(plt.axes(color_axes), "Color by Grouping")        
        self.color_button.on_clicked(partial(self.color_by_grouping, "3d"))
        
        self.fig_3d.canvas.mpl_connect("motion_notify_event", 
                                       partial(self.plot_hover, "3d"))
        self.fig_3d.canvas.mpl_connect('pick_event', self.plot_pick)
        
        plt.show()
    

    def color_by_grouping(self, plot_type=None, event=None):
        """Lets the user choose the grouping to use for coloring.
        
        Parameters
        ----------
        event: matplotlib.backend_bases.MouseEvent, optional
            The event that triggered the color_by_grouping dialog.
        
        Returns
        -------
        None
        """
        
        # Removes empty groupings and condenses groupings from clustering 
        # (i.e., if there are separate groupings for different matrices) 
        groupings = set()
        for grouping_name in self.data_set.groupings.keys():
            
            if len(self.data_set.groupings[grouping_name]) > 0:
                if "Cluster" in grouping_name:
                    split_name = grouping_name.split(" ")
                    grouping_name = (split_name[0] + " " 
                    + " ".join(split_name[-2:]))
                    
                groupings.add(grouping_name)
                
        groupings.add("None")
        
        grouping, ok = qtw.QInputDialog.getItem(
            self, 'Choose Grouping for Plot Coloring:', 'Grouping', 
            list(groupings), editable=False)
        grouping = str(grouping)
            
        if ok:
            if grouping == "None":
                self.color_grouping = None
            else:
                self.color_grouping = grouping
            
            plt.close()
            
            if plot_type=="3d":
                self.show_3D_MDS()
            elif plot_type=="2d":
                self.show_2D_MDS()
            else:
                self.heat_map()


###############################################################################
# Score visualization methods
            
    def match_visualization_widget(self, plotting_change=False):
        """Creates the widget for displaying match visualizations.
        
        This widget is the basis for the histogram, ROC, and linear ordering 
        displays. it creates the figure, rank table, sliders (if needed), 
        buttons, and all other information besides the content of the figure 
        itself.
        
        Parameters
        ----------
        plotting_change : {False, True}, optional
            Stores whether or not interpolation scores should be maintained.
        
        Returns
        -------
        None
        
        See Also
        --------
        histogram: Plots the histogram on the widget.
        smooth_histogram: Plots a smooth histogram on the widget.
        show_roc_curve: Plots an ROC curve on the widget.
        linear_ordering: Plots a linear ordering of the scores on the widget.
        """
        
        if self.mv_widget:
            self.mv_widget.close()
            
        self.cur_points = []
        self.interpolation_scores = []
            
        self.mv_widget = qtw.QWidget()
        self.mv_widget.setWindowTitle("Binary Classification Performance")
        self.mv_widget.setWindowIcon(self.data_set.icon)
        vbox = qtw.QVBoxLayout(self.mv_widget)
        
        mv_scroll = qtw.QScrollArea(self.mv_widget) 
        mv_scroll.setWidgetResizable(True)        
        mv_scroll_container = qtw.QWidget()
        mv_scroll.setWidget(mv_scroll_container)
        container_layout = qtw.QVBoxLayout()
        mv_scroll_container.setLayout(container_layout)
        
        # Creates upper figure
        self.mv_figure = mpl.figure.Figure()
        self.mv_figure.set_tight_layout({"pad": 3.5})
        self.mv_canvas = FigureCanvas(self.mv_figure)
        self.mv_canvas.setParent(self.mv_widget)
        self.mv_canvas.setFocusPolicy(qtcore.Qt.StrongFocus)
        self.mv_canvas.setFocus()
        self.mv_canvas.setSizePolicy(qtw.QSizePolicy.Preferred,
                                     qtw.QSizePolicy.Preferred)
        
        self.mpl_toolbar = NavigationToolbar(self.mv_canvas, 
                                             mv_scroll_container)
        
        # Adds widget for information below figure
        self.mv_info_widget = qtw.QWidget(mv_scroll_container)
        info_layout = qtw.QGridLayout()
        
        
        # Adds plot display and selector buttons
        buttons_layout = qtw.QHBoxLayout()
        
        one_plot_label = "Show on Separate Plots"
        if not self.use_1_plot:
            one_plot_label = "Show on Single Plot"

        self.one_plot_button = self.gb.make_widget(qtw.QPushButton(
            one_plot_label), "clicked", self.roc_plotting_change)
        self.one_plot_button.setVisible(False)
        buttons_layout.addWidget(self.one_plot_button)
        buttons_layout.addStretch(2)
        info_layout.addLayout(buttons_layout, 0, 0)
        
        font = qtgui.QFont()
        self.stats_tables = []
        self.stats_labels = []
        self.bw_labels = []
        self.bw_sliders = []
        for matrix_index, matrix in enumerate(self.selected_matrices):
            
            # Creates bandwidth slider (visible only for smooth histogram)
            new_bw_slider = qtw.QSlider(qtcore.Qt.Horizontal)
            new_bw_slider.setTickPosition(qtw.QSlider.TicksBelow)
            new_bw_slider.setVisible(False)
            self.bw_sliders.append(new_bw_slider)
            
            font.setPointSize(10)
            new_bw_label = self.gb.make_widget(qtw.QLabel(
                "Bandwidth: " + matrix), font=font)
            new_bw_label.setVisible(False)
            self.bw_labels.append(new_bw_label)
            
            info_layout.addWidget(new_bw_label, 1, matrix_index)
            info_layout.addWidget(new_bw_slider, 2, matrix_index)
            
            # Adds rank table
            font.setBold(True)
            info_layout.addWidget(self.gb.make_widget(qtw.QLabel(
                "Rank Table: " + matrix), font=font), 3, matrix_index)
            
            stats_table = qtw.QTableWidget()
            stats_table.setRowCount(4)
            stats_table.setColumnCount(3)
            hlabels = ["Left (lower)", "Right (higher)", "Total"]
            stats_table.setHorizontalHeaderLabels(hlabels)
            vert_text = ("Known Matches: Quantity;Known Non-Matches: Quantity;"
                         "Known Matches: Percent;Known Non-Matches: Percent")
            stats_table.setVerticalHeaderLabels(vert_text.split(";"))
            stats_table.verticalHeader().setDefaultSectionSize(29)
            
            font.setBold(False)
            font.setPointSize(9)
            stats_label = self.gb.make_widget(qtw.QLabel("Score:"), font=font)
            self.stats_labels.append(stats_label)
            info_layout.addWidget(stats_label, 4, matrix_index)
            self.stats_tables.append(stats_table)
            info_layout.addWidget(stats_table, 5, matrix_index)

        
        self.mv_canvas.mpl_connect("motion_notify_event", self.mv_hover)
        self.mv_canvas.mpl_connect('button_press_event', self.mv_click)
        self.mv_canvas.mpl_connect('pick_event', self.mv_pick)
        
        
        self.mv_info_widget.setLayout(info_layout)
        self.mv_info_widget.setMaximumHeight(250)                               #new
        
        container_layout.addWidget(self.mv_canvas, 2)
        container_layout.addWidget(self.mpl_toolbar)
        container_layout.addWidget(self.mv_info_widget)
        
        vbox.addWidget(mv_scroll)
        self.mv_widget.setLayout(vbox)
        
        if not plotting_change:
            self.interpolation_scores = []
            
            
    def histogram(self):
        """Displays a histogram of the data.
        
        The default bin size is the square root of the number of scores.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        See Also
        --------
        match_visualization_widget: Creates the widget on which the figure is 
            plotted.
        mv_hover: Displays data when hovering over the plot.
        
        """
        
        if len(self.selected_matrices) == 0:
            self.gb.no_selection_error()
            return
            
        self.match_visualization_widget()    

        self.current_axes = [None for axis in range(len(
            self.selected_matrices))]
            
        self.mv_figure.suptitle("Data Histogram", fontsize=18)
        
        for matrix_index, matrix_name in enumerate(self.selected_matrices):
            
            self.current_axes[matrix_index] = self.mv_figure.add_subplot(
                1, len(self.selected_matrices), matrix_index+1) 
            
            mnm_scores = self.data_set.match_nonmatch_scores[matrix_name]
            match_scores = [row[0] for row in mnm_scores["matches"]]
            nonmatch_scores = [row[0] for row in mnm_scores["nonmatches"]]
            all_scores = [row[0] for row in mnm_scores["all"]]
            
            bin_size = int(math.sqrt(len(match_scores) + len(nonmatch_scores)))
            nonmatch_weights = np.zeros_like(nonmatch_scores)
            nonmatch_weights += 1. / len(nonmatch_scores)
            match_weights = np.zeros_like(match_scores) + 1./len(match_scores)
            
            axis = self.current_axes[matrix_index]
            
            axis.hist(nonmatch_scores, bin_size, alpha=.5, 
                      weights=nonmatch_weights, label="Unmatched Scores",
                      color="red")
            axis.hist(match_scores, bin_size, alpha=.5, weights=match_weights, 
                         label="Matched Scores", color="blue")
                         
            axis.legend(loc='upper right')
            axis.set_title(matrix_name)
            axis.set_xlabel("Score")
            axis.set_ylabel("Relative Frequency")
            axis.set_xlim(min(all_scores), max(all_scores))
            
            
        self.cursor = MultiCursor(self.current_axes[0].get_figure().canvas, 
                                  self.current_axes, horizOn=False, 
                                  useblit=True, linewidth=1, ls="--", 
                                  color='black')

        self.mv_plot_type = "histogram"
        width = min(self.screen_geometry.width(), 
                    600*len(self.selected_matrices))
        self.mv_widget.setGeometry(40, 70, width, 700)
        self.mv_hover()
        self.mv_widget.show()
        
    
    def smooth_histogram(self, bandwidth=None):
        """Displays a histogram of the data smoothed with Gaussian KDE.  
        
        Parameters
        ----------
        bandwidth : int, optional
            The bandwidth of the smoothed histogram.
        
        Returns
        -------
        None
        
        See Also
        --------
        match_visualization_widget: Creates the widget on which the figure is 
            plotted.
        bandwidth_changed: Adjusts the smooth histogram's bandwidth.
        scipy.stats.gaussian_kde: The method used for smoothing the histogram.
        mv_hover: Displays data when hovering over the plot.
        
        """
    
        if len(self.selected_matrices) == 0:
            self.gb.no_selection_error()
            return
        
        if not bandwidth:   
            self.match_visualization_widget()
            
            m_range = range(len(self.selected_matrices))
            self.current_axes = [None for axis in m_range]
            self.mv_figure.suptitle("Data Histogram", fontsize=18)     
            
        self.cur_points =[]
        for matrix_index, matrix_name in enumerate(self.selected_matrices):
            
            mnm_scores = self.data_set.match_nonmatch_scores[matrix_name]
            match_scores = [row[0] for row in mnm_scores["matches"]]
            nonmatch_scores = [row[0] for row in mnm_scores["nonmatches"]]
            all_scores = [row[0] for row in mnm_scores["all"]]
            max_score = max(all_scores)
        
            x = np.linspace(0, max_score, 200)
    
            if bandwidth:
                self.current_axes[matrix_index].clear()
                gkde_match = stats.gaussian_kde(match_scores, bandwidth)
                gkde_nonmatch = stats.gaussian_kde(nonmatch_scores, bandwidth)
            else:
                self.current_axes[matrix_index] = self.mv_figure.add_subplot(
                    1, len(self.selected_matrices), matrix_index+1) 
                gkde_match = stats.gaussian_kde(match_scores)
                gkde_nonmatch = stats.gaussian_kde(nonmatch_scores)
                self.bandwidth_init = gkde_match.covariance_factor()
                
                # Adds bandwidth slider
                cur_bw = gkde_match.covariance_factor()
                self.bw_sliders[matrix_index].setMinimum(1)
                self.bw_sliders[matrix_index].setMaximum(int(400*cur_bw))
                self.bw_sliders[matrix_index].setValue(int(100*cur_bw))
                self.bw_sliders[matrix_index].setTickInterval(
                    qtw.QSlider.NoTicks)
                self.bw_sliders[matrix_index].setVisible(True)
                self.bw_sliders[matrix_index].sliderReleased.connect(partial(
                    self.bandwidth_changed, matrix_index))
                
                self.bw_labels[matrix_index].setText(
                    "Bandwidth: " + str(format(cur_bw, '.4g')))
                self.bw_labels[matrix_index].setVisible(True)
            
            total = len(all_scores)
            
            matches_relative = [point/total for point in gkde_match(x)]
            nonmatches_relative = [point2/total for point2 in gkde_nonmatch(x)]
            self.cur_points.append([matches_relative, nonmatches_relative])
            
            # Adds separate labels
            # (fill_between labels don't work before mpl 1.5.0)
            axis = self.current_axes[matrix_index]
            axis.plot([0],[0], color="blue", label='Matched Scores')                   
            axis.plot([0],[0], color="red", label='Unmatched Scores')
            axis.fill_between(x, 0, matches_relative, facecolor="blue", 
                              alpha=.5, label='Match')
            axis.fill_between(x, 0, nonmatches_relative, facecolor="red", 
                              alpha=.5, label='Non-Match')
            axis.set_xlim(0, max_score)
            
            max_y = 1.1*max([max(matches_relative), max(nonmatches_relative)])
            axis.set_ylim(0, max_y)
            
            axis.legend(loc='upper right')
            axis.set_title(matrix_name)
            axis.set_xlabel("Score")
            axis.set_ylabel("Relative Frequency")

        self.cursor = self.CustomCursor(
            self.current_axes[0].get_figure().canvas, self.current_axes, 
            useblit=True, horizOn=True, points=self.cur_points, 
            mnm=self.data_set.match_nonmatch_scores, selected_matrices = 
            self.selected_matrices, linewidth=1, ls="--", color='black')
                
        self.mv_plot_type = "smooth histogram"
        width = min(self.screen_geometry.width(), 
                    600*len(self.selected_matrices))
        height = min(self.screen_geometry.height(), 850)
        self.mv_widget.setGeometry(40, 70, width, height)
        self.mv_hover()
        self.mv_widget.show()
            
    
    def bandwidth_changed(self, matrix_index):
        """Updates the smooth histogram's bandwidth.  
        
        Parameters
        ----------
        matrix_index : int
            The index of the plot whose bandwidth is being changed.
        
        Returns
        -------
        None
        
        See Also
        --------
        smooth_histogram: Creates the smooth histogram.
        match_visualization_widget: Creates the widget on which the figure is 
            plotted.
        scipy.stats.gaussian_kde: The method used for smoothing the histogram.
        mv_click: Handles clicking on the plot and labelling data.
        mv_hover: Displays data when hovering over the plot.
        
        """

        bandwidth = self.bw_sliders[matrix_index].value()
        bandwidth = float(bandwidth)/100
        
        matrix_name = self.current_axes[matrix_index].get_title()
        
        mnm_scores = self.data_set.match_nonmatch_scores[matrix_name]
        match_scores = [row[0] for row in mnm_scores["matches"]]
        nonmatch_scores = [row[0] for row in mnm_scores["nonmatches"]]
        all_scores = [row[0] for row in mnm_scores["all"]]
        max_score = max(all_scores)
        
        gkde_match = stats.gaussian_kde(match_scores, bandwidth)
        gkde_nonmatch = stats.gaussian_kde(nonmatch_scores, bandwidth)
        
        total = len(all_scores)
        
        x = np.linspace(0, max_score, 200)
        matches_relative = [point / total for point in gkde_match(x)]
        nonmatches_relative = [point2 / total for point2 in gkde_nonmatch(x)]
        self.cur_points[matrix_index] = [matches_relative, nonmatches_relative]
        
        # Draws the new histograms
        axis = self.current_axes[matrix_index]
        axis.clear()
        
        axis.plot([0],[0], color="blue", label='Matched Scores')                   
        axis.plot([0],[0], color="red", label='Unmatched Scores')
        axis.fill_between(x, 0, matches_relative, facecolor="blue", alpha=.5, 
                          label='Match')
        axis.fill_between(x, 0, nonmatches_relative, facecolor="red", alpha=.5,
                          label='Non-Match')
        axis.set_xlim(0, max_score)
        
        axis.legend(loc='upper right')
        axis.set_title(matrix_name)
        axis.set_xlabel("Score")
        axis.set_ylabel("Relative Frequency")
        axis.get_figure().canvas.draw()
        self.bw_labels[matrix_index].setText(
            "Bandwidth: " + str(format(bandwidth, '.4g')))
        
        for score in self.interpolation_scores:
            self.mv_click(prev_score=score)
        
           
    def show_roc_curve(self, plotting_change=False):
        """Creates ROC curves for the score matrices.
        
        The Receiver Operating Characteristic curve plots the true and false 
        positive rates of the data against each other. Hovering over a score 
        gives its true and false positive rates; clicking pins them on the 
        plot.      
        
        Parameters
        ----------
        plotting_change : {False, True}
            Determines whether or not the number of plots was changed (if so, 
            redraws interpolation scores)
        
        Returns
        -------
        None
        
        See Also
        --------
        match_visualization_widget: Creates the widget on which the figure is 
            plotted.
        mv_click: Functionality for clicking on the plot and labelling data.
        roc_plotting_change: Switches between showing the ROC curves on one 
            plot or separate plots.
        """
        
        if len(self.selected_matrices) == 0:
            self.gb.no_selection_error()
            return
        
        self.mv_plot_type = "roc"    
        self.match_visualization_widget(plotting_change=True)    
        
        if not plotting_change:
            self.interpolation_scores = []

        if len(self.selected_matrices) > 1:
            self.one_plot_button.setVisible(True)

        num_plots = len(self.selected_matrices)
        self.current_axes = [None for axis in range(num_plots)]
        if self.use_1_plot:
            num_plots = 1
            self.current_axes = [None]
            self.current_axes[0] = self.mv_figure.add_subplot(1, 1, 1)
        
        self.mv_figure.suptitle("Receiver Operating Characteristic Curve", 
                                fontsize=18)
             
        
        self.cur_points = []
        self.thresholds = []
        self.artists["mv"] = []
        self.annotations["mv"] = []
        self.selectors = []
        for matrix_index, matrix_name in enumerate(self.selected_matrices):
            
            # Gets ROC from expected and actual scores
            mnm_scores = self.data_set.match_nonmatch_scores[matrix_name]
            flat_scores = [row[0] for row in mnm_scores["all"]]
            gt = mnm_scores["gt"]
            false_pos_rate, true_pos_rate, thresholds = roc_curve(
                gt, flat_scores)
            roc_auc = auc(false_pos_rate, true_pos_rate)
            
            # Creates plots
            plot_label = "AUC = %0.3f" % roc_auc
            if self.use_1_plot:
                axis = self.current_axes[0]
                plot_label = matrix_name + "\n" + plot_label
                
            else:
                self.current_axes[matrix_index] = self.mv_figure.add_subplot(
                    1, len(self.selected_matrices), matrix_index+1)
                axis = self.current_axes[matrix_index]
                axis.set_title(matrix_name)
            
            # Adjusts marker settings
            cur_color = list(self.color_options.keys())[
                             matrix_index % len(self.color_options)]
            style = self.marker_style_options[self.settings[
                "Marker shape"][0].capitalize()] + "-"
            cur_artist, = axis.plot(
                false_pos_rate, true_pos_rate, style, color=cur_color, 
                markersize=4, label=plot_label, picker=10)
            self.artists["mv"].append(cur_artist)
            
            # Adds annotations
            matrix_notes = []
            for index, threshold in enumerate(thresholds):
                note = axis.annotate(
                        threshold, 
                        xy=(false_pos_rate[index], true_pos_rate[index]), 
                        xytext=(false_pos_rate[index] + .02, 
                                true_pos_rate[index] + .02), 
                        arrowprops=dict(arrowstyle='->'), 
                        bbox=dict(boxstyle="round", fc="w"))
                note.set_visible(False)
                matrix_notes.append(note)
            self.annotations["mv"].append(matrix_notes)
            
            # Adjusts axes
            axis.legend(loc='lower right')
            axis.plot([0,1], [0,1], 'r--')
            axis.set_xlim( [-.01, 1.01] )
            axis.set_ylim( [-.01, 1.01] )
            axis.set_ylabel('True Positive Rate')
            axis.set_xlabel('False Positive Rate')
            
            
            self.thresholds.append(thresholds)
            self.cur_points.append([false_pos_rate, true_pos_rate])
            self.selectors.append(
                RectangleSelector(axis, partial(self.plot_rect, matrix_index)))
        
        # If necessary, redraws interpolation scores
        if plotting_change:
            for cur_score in self.interpolation_scores:
                self.mv_click(prev_score=cur_score)

        self.cursor = MultiCursor(self.current_axes[0].get_figure().canvas, 
                                  self.current_axes, horizOn = True, 
                                  useblit=True, linewidth=1, 
                                  ls="--", color='black')

        width = min(self.screen_geometry.width(), 
                    550*len(self.selected_matrices))
        self.mv_widget.setGeometry(40, 70, width, 700)
        self.mv_hover()
        self.mv_widget.show()

    
    def roc_plotting_change(self, label):
        """Switches between showing ROC curves on one plot or separate plots.      
        
        Parameters
        ----------
        label : The label on the button clicked ("Show on Single Plot" or 
            "Show on Separate Plots")
        
        Returns
        -------
        None
        
        See Also
        --------
        show_roc_curve: Displays the ROC curve.
        """
            
        label = str(self.one_plot_button.text())
        if label == "Show on Single Plot":
            self.use_1_plot = True
        else:
            self.use_1_plot = False
            
        plt.close()
        self.show_roc_curve(True)
            
        
    def linear_ordering(self):
        """Creates linear orderings of each matrix's matched & unmatched scores
        
        The upper line displays the scores between matched records; the lower 
        one displays the scores between unmatched records.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        See Also
        --------
        match_visualization_widget: Creates the widget on which the figure is 
            plotted.
        """
        
        if len(self.selected_matrices) == 0:
            self.gb.no_selection_error()
            return
        
        self.mv_plot_type = "linear ordering"
        self.match_visualization_widget()
            
        mtx_range = range(len(self.selected_matrices))
        self.current_axes = [None for axis in mtx_range]
        self.mv_figure.suptitle("Linear Ordering", fontsize=18)
        self.artists["mv"] = []
        self.annotations["mv"] = []
        self.interpolation_scores = []
        style = self.marker_style_options[
            self.settings["Marker shape"][0].capitalize()]
        
        for matrix_index, matrix_name in enumerate(self.selected_matrices):
            self.current_axes[matrix_index] = self.mv_figure.add_subplot(
                len(self.selected_matrices), 1, matrix_index+1) 
            
            mnm_scores = self.data_set.match_nonmatch_scores[matrix_name]
            match_scores = [row[0] for row in mnm_scores["matches"]]
            nonmatch_scores = [row[0] for row in mnm_scores["nonmatches"]]
            all_scores = [row[0] for row in mnm_scores["all"]]
            
            match_y = np.array(match_scores)
            match_y.fill(.005)
            nonmatch_y = np.array(nonmatch_scores)
            nonmatch_y.fill(-.03)
            
            axis = self.current_axes[matrix_index]
            axis.axhline(.005)
            
            # Adds artists
            new_artists = []
            cur_artist, = axis.plot(match_scores, match_y, 'b' + style, 
                      label="Matched Scores", markersize=5, picker=8)
            new_artists.append(cur_artist)        
            axis.axhline(-.03)
            cur_artist, = axis.plot(nonmatch_scores, nonmatch_y, 'r' + style, 
                      label="Unmatched Scores", markersize=5, picker=8)
            new_artists.append(cur_artist) 
            self.artists["mv"].append(new_artists)
            axis.set_xlim(min(all_scores), 
                          max(all_scores))
            axis.set_ylim(-.05, .05)
            axis.yaxis.set_visible(False)
            axis.set_xlabel("Score")
            axis.set_title(matrix_name)
            axis.legend(loc='upper right')
        
            # Adds annotations
            match_notes = []
            for index, match_info in enumerate(mnm_scores["matches"]):
                text = (match_info[1] + " and \n" + match_info[2] + ": \n" 
                        + str(match_info[0]))
                note = self.current_axes[matrix_index].annotate(text,
                    xy=(match_info[0], .005), 
                    xytext=(match_info[0]+.005, .011), 
                    arrowprops=dict(arrowstyle='->'), bbox=dict(
                    boxstyle="round", fc="w"))
                note.set_visible(False)
                match_notes.append(note)
                
            nonmatch_notes = []
            for index, nonmatch_info in enumerate(mnm_scores["nonmatches"]):
                text = (nonmatch_info[1] + " and \n" + nonmatch_info[2] 
                        + ": \n" + str(nonmatch_info[0]))
                note = self.current_axes[matrix_index].annotate(text,
                    xy=(nonmatch_info[0], -.03), 
                    xytext=(nonmatch_info[0], -.05), 
                    arrowprops=dict(arrowstyle='->'), 
                    bbox=dict(boxstyle="round", fc="w"))
                note.set_visible(False)
                nonmatch_notes.append(note)
                
            self.annotations["mv"].append([match_notes, nonmatch_notes])
            
        # Adjusts plot style
        bottom_height = .2/len(self.selected_matrices)
        self.mv_figure.subplots_adjust(bottom=bottom_height)                                 
        
        self.cursor = MultiCursor(self.current_axes[0].get_figure().canvas, 
                                  self.current_axes, horizOn=False, 
                                  useblit=True, linewidth=1, ls="--", 
                                  color='black')
        
        height = min(self.screen_geometry.height(), 
                     600*len(self.selected_matrices))
        self.mv_widget.setGeometry(40, 70, 1500, height)
        self.mv_hover()
        self.mv_widget.show()
        
        
    def show_stats(self):
        """ Displays the mean, median, and standard deviation for each matrix.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
        info_text = ""
        for matrix_name in self.data_set.matrices:
            cur_matrix = self.data_set.matrices[matrix_name]["matrix"]
            cur_matrix.flatten()
            cur_matrix=cur_matrix[~np.isnan(cur_matrix)]
            
            m_type = self.data_set.matrices[matrix_name]["type"].lower()
            info = (m_type, np.average(cur_matrix), m_type, 
                    np.median(cur_matrix), np.std(cur_matrix))
            
            info_text += (matrix_name + ": \n\n" + ("  Mean %s: %s \n "
                " Median %s: %s \n  Standard deviation: %s" %info) 
                + "\n\n")
        
        self.gb.general_msgbox(self.data_set.name, 
                               "%s Statistics:" % self.data_set.name, 
                               info_text, qtw.QMessageBox.Information)
                                        

###############################################################################
# Interpolation methods

    def get_interpolation_score(self):
        """ Gets score to interpolate from user.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        score : float
            The score entered by the user.
        
        """
        
        score, ok = qtw.QInputDialog.getText(self, 'Score Interpolation', 
                                             'Enter score:')
        if not ok:
            return False
        try:
            score = float(score)
        except ValueError:
            self.gb.general_msgbox("Error", "Please enter a number.")
            return self.get_interpolation_score()
            
        self.interpolation_scores.append(score)
        return score          

###############################################################################
# Clustering methods

    def cluster(self):
        """ Plots the results of the chosen clustering method.
        
        Clusters are stored in `self.clusters`, and the items in each cluster 
        are listed under the plot.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        See Also
        --------
        cluster_options: Opens a widget that lets the user pick 
            clustering options.
        view_clusters: Displays the records in each cluster.
        hierarchical_clustering: Performs hierarchical clustering on the data.
        spectral_clustering: Performs spectral clustering on the data.
        """
        
        if self.options_widget:
            self.options_widget.close()
        
        # Sets up the figure and canvas
        self.cluster_widget = qtw.QWidget()
        self.cluster_widget.setWindowTitle("Clustering")
        self.cluster_widget.setWindowIcon(self.data_set.icon)
        
        vbox = qtw.QVBoxLayout(self.cluster_widget)
        cluster_scroll = qtw.QScrollArea(self.cluster_widget) 
        cluster_scroll.setWidgetResizable(True)        
        cluster_scroll_container = qtw.QWidget()
        cluster_scroll.setWidget(cluster_scroll_container)
        container_layout = qtw.QVBoxLayout()
        cluster_scroll_container.setLayout(container_layout)
        
        self.cluster_figure = mpl.figure.Figure()
        self.cluster_canvas = FigureCanvas(self.cluster_figure)
        self.cluster_canvas.setParent(self.cluster_widget)
        self.cluster_canvas.setFocusPolicy(qtcore.Qt.StrongFocus)
        self.cluster_canvas.setFocus()
        self.cluster_canvas.setSizePolicy(qtw.QSizePolicy.Preferred,
                                          qtw.QSizePolicy.Preferred)
        
        self.mpl_toolbar = NavigationToolbar(self.cluster_canvas, 
                                             cluster_scroll_container)
        
        # Calculates and stores clusters for current clustering method
        self.labels = None
        self.cur_points = []
        self.cluster_methods[str(self.cluster_method)]()

        # Adds widget for buttons and labels below figure
        self.info_widget = qtw.QWidget(cluster_scroll_container)
        info_layout = qtw.QGridLayout()
        
        widget_index = 1
        
        buttons_layout = qtw.QHBoxLayout()
        buttons_layout.addWidget(self.gb.make_widget(qtw.QPushButton(
            " Clustering Preferences "), "clicked", self.cluster_options))

        if self.labels is not None:

            self.lasso_btn = self.gb.make_widget(qtw.QPushButton(
                "Lasso Select"), "clicked", 
                partial(self.cluster_selector_change, "cluster", rect=False))
            self.rect_btn = self.gb.make_widget(qtw.QPushButton(
                "Rectangle Select"), "clicked", 
                partial(self.cluster_selector_change, "cluster", rect=True))
            self.rect_btn.setDefault(True)
            
            font = qtgui.QFont()
            font.setPointSize(11)
            font.setBold(True)
            
            buttons_layout.addWidget(self.gb.make_widget(qtw.QPushButton(
                "View Records in All Clusters"), "clicked", 
                self.view_clusters))
            info_layout.addWidget(self.gb.make_widget(qtw.QLabel("Clusters"), 
                                  font=font), 0, 0)
            lasso_rect_layout = qtw.QHBoxLayout()
            lasso_rect_layout.addStretch(2)
            lasso_rect_layout.addWidget(self.lasso_btn)
            lasso_rect_layout.addWidget(self.rect_btn)
            info_layout.addLayout(lasso_rect_layout, 0, 1, 1, 2)
            font.setPointSize(9)
            
            # For each matrix, lists items in each cluster, if possible.
            for index, matrix_name in enumerate(self.selected_matrices):
                info_layout.addWidget(self.gb.make_widget(qtw.QLabel(
                    "Clusters for " + matrix_name + ":"), font=font), 
                    widget_index, 0, 1, 2)
                widget_index += 1

                # Displays lists of items in each cluster
                keys=list(self.clusters[matrix_name].keys())
                keys.sort()
                for label in keys:
                    cur_label = qtw.QLabel("Cluster " + str(label+1) + ": ")
                    cur_edit = qtw.QLineEdit(", ".join(
                        self.clusters[matrix_name][label]))
                    cur_edit.setReadOnly(True)
                    info_layout.addWidget(cur_label, widget_index, 0)
                    info_layout.addWidget(cur_edit, widget_index, 1)
                    
                    info_layout.addWidget(self.gb.make_widget(qtw.QPushButton(
                        "View"), "clicked", partial(self.view_clusters, 
                        matrix_name, label)), widget_index, 2)
                    
                    widget_index += 1 

            self.cluster_canvas.mpl_connect(
                "motion_notify_event", partial(self.plot_hover, "cluster"))
            self.cluster_canvas.mpl_connect('pick_event', self.plot_pick)
                    
        else:
            self.cluster_canvas.mpl_connect("motion_notify_event",
                                            self.hier_hover)
            self.cluster_canvas.mpl_connect('pick_event', self.hier_pick)

        info_layout.addLayout(buttons_layout, widget_index+1, 1)
        self.info_widget.setLayout(info_layout)
        
        container_layout.addWidget(self.cluster_canvas, 2)
        container_layout.addWidget(self.mpl_toolbar)
        container_layout.addWidget(self.info_widget)
        
        vbox.addWidget(cluster_scroll)
        self.cluster_widget.setLayout(vbox)
        self.cluster_widget.show()
        
    
    def cluster_options(self):
        """ Opens a widget that lets the user pick clustering options.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        See Also
        --------
        cluster: Plots the results of the chosen clustering method.
        cluster_method_choice: Updates preferences when the user chooses a 
            clustering method. 
        VisualMetricAnalyzer class: More details on the options for each 
            clustering method.
        hierarchical_clustering: Performs hierarchical clustering on the data.
        spectral_clustering: Performs spectral clustering on the data.
            
        """
        
        if self.cluster_widget:
            self.cluster_widget.close()
        
        # Adds widget for clustering options
        self.options_widget = qtw.QWidget()
        self.options_widget.setWindowTitle("Clustering Options")
        self.options_widget.setWindowIcon(self.data_set.icon)
        
        options_layout = qtw.QGridLayout()
        self.options_widget.setLayout(options_layout)

        font = qtgui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        
        title_label = self.gb.make_widget(qtw.QLabel("Clustering Options"), 
                                          font=font)
        title_label.setFixedHeight(50)
        options_layout.addWidget(title_label, 0, 0)
        options_layout.setSpacing(10)

        method_items = list(self.cluster_methods.keys())                        
        cur_method = method_items.index(str(self.cluster_method))
            
        self.method_list = self.gb.make_widget(
            qtw.QComboBox(), "currentIndexChanged", 
            self.cluster_method_choice, items=method_items)
          
        self.method_prefs_layout = qtw.QGridLayout()
        self.method_prefs_layout.setVerticalSpacing(10)
        method_prefs_box = qtw.QGroupBox("Preferences")
        method_prefs_box.setLayout(self.method_prefs_layout)

        options_layout.addWidget(self.method_list, 1, 0)
        options_layout.addWidget(method_prefs_box, 1, 1, 3, 3)
        options_layout.addWidget(self.gb.make_widget(
            qtw.QPushButton("Done"), "clicked", self.cluster), 4, 3)
        self.method_list.setCurrentIndex(cur_method)
        self.cluster_method_choice(self.method_list.findText(
            self.cluster_method))
        self.options_widget.show()


    def cluster_method_choice(self, item):
        """  Updates preferences when the user chooses a clustering method. 
        
        Parameters
        ----------
        item : QComboBoxItem
            The item selected on `options_widget`.
        
        Returns
        -------
        None
        
        See Also
        --------
        cluster_options: Creates the widget that lets the user pick clustering 
            options.
        VisualMetricAnalyzer class: More details on the options for each 
            clustering method.
        """
        self.cluster_method = self.method_list.itemText(item)

        # Removes old clustering preferences
        for layout_index in reversed(range(len(self.method_prefs_layout))):
            
            item = self.method_prefs_layout.itemAt(layout_index)
            self.method_prefs_layout.removeItem(item)
            item.widget().setParent(None)
        
        # Adds hierarchical clustering preferences
        if self.cluster_method == "Hierarchical":
            
            hier_cluster_items = ["Single", "Complete", "Average", "Weighted", 
                                  "Centroid", "Median", "Ward"]
            cur_pref = str(self.hier_prefs["method"][0]).title()
            cur_index = hier_cluster_items.index(cur_pref)

            self.method_prefs_layout.addWidget(self.gb.make_widget(
                qtw.QComboBox(), "currentIndexChanged", self.update_pref, 
                items=hier_cluster_items, index=cur_index,
                add_to=self.hier_prefs["method"], is_partial=True), 0, 1)
            self.method_prefs_layout.addWidget(self.gb.make_widget(
                qtw.QLabel("Method:"), add_to=self.hier_prefs["method"]), 
                0, 0)
            
            trunc_items = ["No truncation","Last P clusters", "Up to P levels"]
            self.method_prefs_layout.addWidget(self.gb.make_widget(
                qtw.QComboBox(), "currentIndexChanged", self.update_pref, 
                items=trunc_items, add_to=self.hier_prefs["truncate_mode"], 
                is_partial=True), 1, 1)
            self.method_prefs_layout.addWidget(self.gb.make_widget(
                qtw.QLabel("Truncation:"), 
                add_to=self.hier_prefs["truncate_mode"]), 1, 0)
    
            self.method_prefs_layout.addWidget(self.gb.make_widget(
                qtw.QSpinBox(), "valueChanged", self.update_pref, value=10, 
                minimum=1, add_to=self.hier_prefs["p"], is_partial=True), 2, 1)
            self.method_prefs_layout.addWidget(self.gb.make_widget(
                qtw.QLabel("P:"), add_to=self.hier_prefs["p"]), 2, 0)

                
        # Adds spectral clustering preferences    
        elif self.cluster_method == "Spectral":
            self.method_prefs_layout.addWidget(self.gb.make_widget(
                qtw.QSpinBox(), "valueChanged", self.update_pref, 
                value=self.spectral_prefs["n_clusters"][0], minimum=1, 
                add_to=self.spectral_prefs["n_clusters"], is_partial=True), 
                0, 1)
            self.method_prefs_layout.addWidget(self.gb.make_widget(
                qtw.QLabel("Clusters:"), 
                add_to=self.spectral_prefs["n_clusters"]), 0, 0)
    
            self.method_prefs_layout.addWidget(self.gb.make_widget(
                qtw.QSpinBox(), "valueChanged", self.update_pref, 
                value=self.spectral_prefs["n_clusters"][0], minimum=1, 
                add_to=self.spectral_prefs["n_components"], is_partial=True), 
                1, 1)
            self.method_prefs_layout.addWidget(self.gb.make_widget(
                qtw.QLabel("Components:"), 
                add_to=self.spectral_prefs["n_components"]), 1, 0)
                                     
            self.method_prefs_layout.addWidget(self.gb.make_widget(
                qtw.QSpinBox(), "valueChanged", self.update_pref, value=10, 
                minimum=1, add_to=self.spectral_prefs["n_init"], 
                is_partial=True), 2, 1)
            self.method_prefs_layout.addWidget(self.gb.make_widget(
                qtw.QLabel("Runs:"), add_to=self.spectral_prefs["n_init"]), 
                2, 0)                   
            self.method_prefs_layout.addWidget(self.gb.make_widget(
                qtw.QDoubleSpinBox(), "valueChanged", self.update_pref, 
                value=0.0, minimum=1, add_to=self.spectral_prefs["eigen_tol"], 
                is_partial=True), 3, 1)
            self.method_prefs_layout.addWidget(self.gb.make_widget(
                qtw.QLabel("Eigen tolerance:"), 
                add_to=self.spectral_prefs["eigen_tol"]), 3, 0)
                
        self.options_widget.setSizePolicy(qtw.QSizePolicy.Minimum,
                                          qtw.QSizePolicy.Minimum)
    
    def view_clusters(self, matrix_name=None, cluster_name=None):
        """Displays the records in each cluster.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        See Also
        --------
        cluster: Creates the plot displaying the results of the chosen 
            clustering method.
        view_selected_records: Displays the records.
        
        """
        
        if len(self.selected_data_types) == 0:
            if not self.data_set.has_files:
                self.gb.general_msgbox(
                    "Data Files Unavailable", 
                    "No files are available for these records.")
            else:
                self.gb.general_msgbox("No Data Types Selected", ("Please"
                                            " check the boxes of the data "
                                            "types you would like to view."))
            return       
        
        
        if matrix_name:
            
            self.display_title = matrix_name +" Cluster #%d"%(cluster_name + 1)
            items = self.clusters[matrix_name][cluster_name]
            self.selected_records = []
            for cur_id in items:
                cur_record = self.data_set.record_from_id(cur_id)
                self.selected_records.append(cur_record)
                
            self.view_selected_records()
                
        else:
                
            for matrix_name in self.clusters:            
                for cluster_name in self.clusters[matrix_name]:
                    items = self.clusters[matrix_name][cluster_name]
                
                    self.display_title = (matrix_name + 
                                          " Cluster #%d" % (cluster_name + 1))
                    
                    self.selected_records = []
                    for cur_id in items:
                        cur_record = self.data_set.record_from_id(cur_id)
                        self.selected_records.append(cur_record)
                        
                    self.view_selected_records()
    

    def store_cluster_groupings(self):
        """Stores a new grouping of scores according to the current clustering.
        
        Grouping names have the format "[index]: [matrix] [method] Clustering" 
        and group names have the format 
        "[index]: [matrix] [method] Cluster [cluster number]".
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        See Also
        --------
        cluster: Creates the plot displaying the results of the chosen 
            clustering method.
        """
        

        for matrix_name in self.clusters:
            
            grouping_title = str(self.cluster_grouping_index) + str(
                ": " + matrix_name + " " + self.cluster_method + " Clustering")
            self.data_set.groupings[grouping_title] = {}
            
            for cluster_name in self.clusters[matrix_name]:
                
                group_title = grouping_title[:-3] + " " + str(cluster_name + 1)
                
                new_group = []
                for cur_id in self.clusters[matrix_name][cluster_name]:
                    cur_record = self.data_set.record_from_id(cur_id)
                    
                    cur_record.groups.append(group_title)
                    new_group.append(cur_id)
                    
                self.data_set.groupings[grouping_title][group_title]=new_group
            
        self.cluster_grouping_index += 1

    
    def update_pref(self, to_update, value=None, values=None):
        """Updates the user's clustering preferences.
        
        Given a selection in `options_widget`, updates `cluster_method`, 
        `hier_prefs`, or `spectral_prefs` appropriately.
        
        Parameters
        ----------
        to_update : dict
            The dict of options in which `value` needs to be updated.
        value : str, optional
            The option selected that needs to be updated in `to_update`.
        values : list, optional
            If applicable, the list of options in which `value` occurs.
            
        Returns
        -------
        None
        
        See Also
        --------
        cluster_options: Creates the widget that lets the user pick clustering 
            options.
        VisualMetricAnalyzer class: More details on the options for each 
            clustering method.
        """
        
        if values != None:
            value = values[value]
            
        if value == "No truncation":
            value = None
        elif value == "Last P clusters":
            value = "lastp"
        elif value == "Up to P levels":
            value = "level"
        
        if isinstance(value, str):
            value = value.lower()
            
        to_update[0] = value
        
    
    def hierarchical_clustering(self):
        """Performs hierarchical clustering on the data.
        
        Creates a dendrogram of the data, labels nodes with the scores to which 
        they correspond, and stores the root of the tree for each matrix in 
        `self.roots`.
        `self.tree_coords` is a list of nodes and their x and y values.
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        
        See Also
        --------
        cluster: Plots the results of the chosen clustering method.
        """

        if len(self.selected_matrices) == 0:
            self.gb.no_selection_error()
            return
        
        self.cluster_figure.suptitle("Hierarchical Clustering", fontsize=18)
        self.current_cluster_axes=[None for axis in range(
            len(self.selected_matrices))]
        
        self.tree_coords = {}
        self.roots = {}
        self.annotations["cluster"] = []
        self.artists["cluster"] = []
        for m_index, matrix_name in enumerate(self.selected_matrices):
            
            self.current_cluster_axes[m_index]=self.cluster_figure.add_subplot(
                len(self.selected_matrices), 1, m_index+1) 
            
            cur_matrix = self.data_set.matrices[matrix_name]["matrix"]
            num_matrix = np.nan_to_num(cur_matrix)
            hier_method = self.hier_prefs["method"][0].lower()
            linkage_matrix = linkage(num_matrix, hier_method)
            
            # Makes dendrogram
            self.current_cluster_axes[m_index].set_title(matrix_name)
            self.current_cluster_axes[m_index].set_xlabel("IDs")
            self.current_cluster_axes[m_index].set_ylabel(
                self.data_set.matrices[matrix_name]["type"])
            
            args = self.get_cluster_args(dendrogram, self.hier_prefs, 
                                         [linkage_matrix], 
                                         self.current_cluster_axes[m_index])

            ddata=dendrogram(*args)
            
            self.roots[matrix_name] = to_tree(linkage_matrix)
            self.tree_coords[matrix_name] = [] # stores x, y, node
            xvals = ddata['icoord'] # icoords correspond to x values
            heights = [item[1] for item in ddata['dcoord']] # dcoords=y values
            
            # Non-singleton clusters start here (e.g., if there are 100 
            # records, records 0-99 correspond to original nodes; nodes 100+ 
            # correspond to tree nodes that combine those records)
            total_ids = len(cur_matrix)
            indexed_heights = np.array(
                [[num for num in range(len(heights))], heights])
            
            # Stores each node and the corresponding coordinates. The lowest y 
            # value goes with the first node, second-lowest with the second,...
            matrix_annotations = []
            leaf_annotations = []
            blue_coords = []
            for cur_id in range(total_ids, 2*total_ids-1):
                mincol = np.argmin(indexed_heights[1]) # index of highest y val
                minindex = int(indexed_heights[0][mincol])
                ymin = indexed_heights[1][mincol] # highest y value
                
                # corresponding x value 
                # (average coordinates of left and right subtree branches)
                x = .5 * (xvals[minindex][1] + xvals[minindex][3]) 
    
                self.tree_coords[matrix_name].append([cur_id, x, ymin])
                indexed_heights = np.delete(indexed_heights, mincol, 1)
                
                
                # Adds annotations
                parent = self.get_node_with_id(self.roots[matrix_name], cur_id)
                children = [self.data_set.records[child].id 
                            for child in parent.pre_order()]
                node_text = "\n".join(wrap("IDs: " + ", ".join(children), 30))
                note = self.current_cluster_axes[m_index].annotate(
                    node_text, xy=(x, ymin), arrowprops=dict(arrowstyle='->'), 
                    bbox=dict(boxstyle="round", fc="w"))
                note.draggable()
                note.set_visible(False)
                matrix_annotations.append(note)
                
                if len(children) == 2:
                    blue_coords.append(xvals[minindex][1])
                    blue_coords.append(xvals[minindex][3])
                    
                    note1 = self.current_cluster_axes[m_index].annotate(
                        children[0], xy=(xvals[minindex][1], 0), 
                        arrowprops=dict(arrowstyle='->'), 
                        bbox=dict(boxstyle="round", fc="w"))
                    note1.draggable()
                    note1.set_visible(False)
                    
                    note2 = self.current_cluster_axes[m_index].annotate(
                        children[1], xy=(xvals[minindex][3], 0), 
                        arrowprops=dict(arrowstyle='->'), 
                        bbox=dict(boxstyle="round", fc="w"))
                    note2.draggable()
                    note2.set_visible(False)
                    
                    leaf_annotations.append(note1)
                    leaf_annotations.append(note2)
                    
            self.annotations["cluster"].append(matrix_annotations)
            self.annotations["cluster"].append(leaf_annotations) 
                
            # Adds red circles to tree nodes
            circle_xs = [coord[1] for coord in self.tree_coords[matrix_name]]
            circle_ys = [coord[2] for coord in self.tree_coords[matrix_name]]
            new_artist, = self.current_cluster_axes[m_index].plot(
                circle_xs, circle_ys, 'ro', markersize=4, picker=5)
            self.artists["cluster"].append(new_artist)
            
            blue_artist, = self.current_cluster_axes[m_index].plot(
                blue_coords, np.zeros(len(blue_coords)), 'bo', markersize=4, 
                picker=5)
            self.artists["cluster"].append(blue_artist)

        height = min(self.screen_geometry.height(), 
                     500*len(self.selected_matrices))
        self.cluster_widget.setGeometry(40, 40, 1500, height)
        
                    
    def get_node_with_id(self, root, cur_id):
        """Given an id, recursively finds and returns that node in the tree.        
        
        Parameters
        ----------
        root : ClusterNode
            The root of the subtree to search.
        cur_id : int
            The ClusterNode ID to search for (not a record ID).
            
        Returns
        -------
        None
        
        See Also
        --------
        cluster: Plots the results of the chosen clustering method.
        
        """

        if root is None or root.get_id() == cur_id:
            return root
            
        left_node = self.get_node_with_id(root.get_left(), cur_id)
        right_node = self.get_node_with_id(root.get_right(), cur_id)

        if left_node is not None:
            return left_node    
        elif right_node is not None:
            return right_node
            
        return None
            
    
    def show_spectral_clustering(self):
        """Performs spectral clustering on the data.
        
        Clusters are stored in `self.clusters`.
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        
        See Also
        --------
        cluster: Plots the results of the chosen clustering method.
        """
        
        if len(self.selected_matrices) == 0:
            self.gb.no_selection_error()
            return
        
        self.mv_plot_type = "spectral"    
        self.cluster_figure.suptitle("Spectral Clustering", fontsize=18)
        self.current_cluster_axes = [None for axis in range(
            len(self.selected_matrices))]

        self.labels = {}
        self.clusters = {}
        self.annotations["cluster"] = []
        self.artists["cluster"] = []
        self.cur_points = []
        self.selectors = []
        cur_marker = self.marker_style_options[
            self.settings["Marker shape"][0].capitalize()]
        for m_index, matrix_name in enumerate(self.selected_matrices):

            self.current_cluster_axes[m_index]=self.cluster_figure.add_subplot(
                1, len(self.selected_matrices), m_index+1) 
            
            cur_matrix = self.data_set.matrices[matrix_name]["matrix"]
            score_matrix0 = np.nan_to_num(np.array(cur_matrix))
            score_matrix = np.maximum(score_matrix0, score_matrix0.T)

            similarity_matrix = score_matrix.max() - score_matrix
            
            args = self.get_cluster_args(
                spectral_clustering, self.spectral_prefs, [similarity_matrix], 
                self.current_cluster_axes[m_index])
    
            self.labels[matrix_name] = spectral_clustering(*args)

            # Stores clusters
            self.clusters[matrix_name] = {}
            for item_index, item in enumerate(self.labels[matrix_name]):
                if not item in self.clusters[matrix_name]:
                    self.clusters[matrix_name][item] = []
                self.clusters[matrix_name][item].append(
                    self.data_set.records[item_index].id) 

            # Graphs the clusters
            mds2 = MDS(2, dissimilarity='precomputed')
            all_pts = mds2.fit_transform(score_matrix) #NEW
            x, y = all_pts.T
            self.cur_points.append([x, y])
            self.current_cluster_axes[m_index].set_title(
                matrix_name + " (MDS view)")

            self.annotations["cluster"].append({})
            self.artists["cluster"].append({})
            for cluster_id in range(self.spectral_prefs["n_clusters"][0]):
                cl_x, cl_y = all_pts[self.labels[matrix_name]==cluster_id,:].T
                cur_artist, = self.current_cluster_axes[m_index].plot(
                    cl_x, cl_y, marker=cur_marker, ls="", picker=10)
                
                # Adds annotations
                group_annotations = []
                for item in self.clusters[matrix_name][cluster_id]:
                    cur_index = self.data_set.index_from_id(item)
                    note_y = y[cur_index] + .05*(max(y)-min(y))
                    note = self.current_cluster_axes[m_index].annotate(
                        item, xy=(x[cur_index], y[cur_index]), 
                        xytext=(x[cur_index], note_y), 
                        arrowprops=dict(arrowstyle='->'), bbox=dict(
                        boxstyle="round", fc="w"))
                    note.set_visible(False)
                    group_annotations.append(note)
                
                self.annotations[
                        "cluster"][m_index][cluster_id] = group_annotations
                self.artists[
                        "cluster"][m_index][cluster_id] = cur_artist
                
            self.selectors.append(RectangleSelector(
                self.current_cluster_axes[m_index],
                partial(self.plot_rect, m_index))) 
            
        window_x = min(self.screen_geometry.width(), 
                       550*len(self.selected_matrices) + 50)
        window_y = min(self.screen_geometry.height(), 
                       40*self.spectral_prefs["n_clusters"][0] + 750)       
        self.cluster_widget.setGeometry(40, 40, window_x, window_y)
        self.store_cluster_groupings()


    def get_cluster_args(self, method, dictionary, initial, axis=None):
        """Returns clustering parameters for a clustering method.
        
        Parameters
        ----------
        method : function
            The method for which to retrieve the parameters.
        dictionary : dict
            The dict of preferences (e.g., `spectral_prefs`) for this 
            clustering method.
        initial : object
            The first parameter of `method`.
        axis : matplotlib.axes.Axes, optional
            The axis on which to plot the clustering, if needed.
            
        Returns
        -------
        parameter_values : list
            The list of parameters for `method`.
        
        See Also
        --------
        show_spectral_clustering: Performs spectral clustering.
        hierarchical_clustering: Performs hierarchical clustering.
        
        """
        
        parameter_names = inspect.getargspec(method)[0]
        parameter_defaults = inspect.getargspec(method)[3]
        
        parameter_values = initial
        for index, name in enumerate(parameter_names[len(initial):]):
            
            to_add = parameter_defaults[index]
            
            if name in dictionary and dictionary[name][0] is not None:
                to_add = dictionary[name][0]
            
            if str(to_add).lower() == "none":
                to_add = None
            
            if name == "ax":
                to_add = axis                
            
            parameter_values.append(to_add)

        return parameter_values

###############################################################################            
    def make_matrices(self):
        """ Generates distance matrices for the data set.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        See Also
        --------
        DataSet.make_matrices: Determines what distance matrices to generate.
        DataSet.generate_matrices: Generates the chosen distance matrices.

        """
        
        self.mtx_options_widget = qtw.QDialog()
        self.data_set.make_matrices(self.mtx_options_widget)
        
        if self.data_set.update == True:
            self.close()
            self.create_main_frame()
            self.data_set.num_widgets_open += 1
            self.show()
            self.data_set.update = False
        
        
    def fuse_matrices(self):
        """Opens a dialog to fuse matrices.
               
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        
        See Also
        --------
        DataSet.make_fused_matrices: Displays options for fusing matrices.
        DataSet.fuse_matrices: Fuses chosen matrices using the selected SVC.
        """

        self.fusion_widget = qtw.QDialog()
        self.data_set.make_fused_matrices(self.fusion_widget)
        
        if self.data_set.update == True:
            self.close()
            self.create_main_frame()
            self.data_set.num_widgets_open += 1
            self.show()
            self.data_set.update = False

          
###############################################################################
# Selection methods

    def table_selection(self):
        """Updates the selected records when the user highlights table items.
        
        The selected records are stored in `selected_records`. If displaying 
        pairs of records, each entry in `selected_records` is a list of IDs; 
        otherwise, each entry is a single ID.
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        
        See Also
        --------
        view_selected_records: Displays the selected records.
        """
        
        cur_table = self.matrix_tabs.currentWidget()
        
        items = cur_table.selectedItems()
        self.selected_records = []

        for item in items:
            
            # List form
            if "and" in cur_table.item(0, 0).text():
            
                row_items = str(cur_table.item(
                    item.row(), 0).text()).split("and")
                record1 = self.data_set.record_from_id(row_items[0].strip())
                record2 = self.data_set.record_from_id(row_items[1].strip())
                new_records = [record1, record2]
                
                if new_records not in self.selected_records:
                    self.selected_records += new_records
                    
            # Table form    
            else:
                row_record = self.data_set.records[item.row()]
                col_record = self.data_set.records[item.column()]
                
                if item.row() != item.column():
                    self.selected_records.append([row_record, col_record])


    def table_pressed(self, item):
        """Displays all selected items when the user right-clicks on the table 
        
        Parameters
        ----------
        item : QTableItem
            The item pressed.
            
        Returns
        -------
        None
        
        See Also
        --------
        view_selected_records: Displays the selected records.
        """ 
        if qtw.QApplication.mouseButtons() == qtcore.Qt.RightButton:
            self.view_selected_records()

###############################################################################
# Record viewing methods

    def view_selected_records(self):
        """Sets up initial figure for record viewing.
        
        Creates one figure per selected data type. `figure_info` stores 
        each figure and its buttons, data type, title, and the index of the 
        first record it displays.
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        
        See Also
        --------
        add_plots: Adds specific plots inside the given figure.
        change_figure: Changes the figure after pressing Next or Previous.
        """
        
        if len(self.selected_records) > 0:
            if len(self.selected_data_types) == 0:
                if not self.data_set.has_files:
                    self.gb.general_msgbox(
                        "Data Files Unavailable", "No files are available for"
                        " these records.")
                else:
                    self.gb.general_msgbox("No Data Types Selected", ("Please"
                                           " check the boxes of the data "
                                           "types you would like to view."))
                return 

            # Create one window per data type.
            fig_info = []

            for index, data_type in enumerate(self.selected_data_types):

                if self.data_set.data_types[data_type]["type"]=="Description":
                    self.show_text(data_type)
                else:
                    
                    dim = math.ceil(math.sqrt(
                        self.settings["Plots per page"][0]))
                    
                    w = min(self.screen_geometry.width(), 380*(dim))
                    h = min(self.screen_geometry.height()-100, 360*(dim))
                    
                    dpi = 150
                    fig_h = max(h/dpi, 
                                4 + dim/3 + .9*len(self.selected_matrices))
                    fig_w = fig_h+1
                    
                    min_size = (200+250*dim, 
                                (140+80*len(self.selected_matrices))*dim)
                    
                    fig_widget = self.gb.ScrollableWidget(
                        self.data_set.fig_index, (100, 20+80*index, w, h), 
                        self.data_set.icon, (fig_w, fig_h), min_size)
                    
                    self.data_set.fig_index += 1
                    buttons = self.add_plots(fig_widget.fig, 0, data_type)
                    
                    # Stores figure info (start=index of first record shown)
                    fig_info.append({"figure": fig_widget.fig, 
                                     "buttons": buttons, "canvas": None, 
                                     "data type": data_type, "start": 0, 
                                     "title": self.display_title, 
                                     "widget": fig_widget})
                    
            self.figure_info[self.selection_id] = fig_info
            self.selection_id += 1

                                                                 
    def add_plots(self, fig, show_start, data_type, title=None):    
        """Adds specific plots inside the given figure.
        
        If displaying individual records, displays each record on a separate 
        plot and labels it with its ID. If displaying pairs of records,
        displays each pair on a separate plot and labels it with their IDs and 
        similarity/dissimilarity scores.
        
        Parameters
        ----------
        fig : matplotlib.pyplot.figure
            The figure in which to add the plots.
        show_start : int
            The index in `selected_records` where the records to display begin.
        data_type : str
            The data type of the files to display.
        title : str, optional
            The title to give to the figure.
            
        Returns
        -------
        bnext : matplotlib.widgets.Button
            The "Next" button at the bottom of the figure.
        bprev : matplotlib.widgets.Button
            The "Previous" button at the bottom of the figure.
            
        See Also
        --------
        GUIBackend.show_curve: Displays a curve file on the plot.
        GUIBackend.show_image: Displays an image file on the plot.
        view_selected_records: Sets up initial figure for record viewing.
        change_figure: Changes the figure after pressing Next or Previous.
        """

        if not title:
            title = self.display_title
        
        show_end = show_start + self.settings["Plots per page"][0]
        records_to_show = self.selected_records[show_start : show_end]
        dim = math.ceil(math.sqrt(min(self.settings["Plots per page"][0], 
                                      len(records_to_show))))
        
        for index, record in enumerate(records_to_show):
            
            records = records_to_show[index]
            if len(records_to_show) == 2:
                axes = fig.add_subplot(1, 2, index+1)
            else:
                axes = fig.add_subplot(dim, dim, index+1)
                
            axes.clear()                                                   
            axes.cla()                                         
            
            style = (self.color_options[
                self.settings["Marker color"][0].capitalize()] + 
                self.marker_style_options[self.settings[
                "Marker shape"][0].capitalize()] + "-")

            # For a 2D list, shows a pair of records.
            if isinstance(records_to_show[index], list):
                
                fig.suptitle("Record Comparison", fontsize=18)
                record1 = records_to_show[index][0]
                record2 = records_to_show[index][1]

                path1 = "Unavailable"
                path2 = "Unavailable"
                if data_type in records[0].files:
                    path1 = self.data_set.get_file_path(record1, data_type)
                if data_type in records[1].files:    
                    path2 = self.data_set.get_file_path(record2, data_type)
                
                if data_type == "Point Cloud":
                    self.gb.show_curve(axes, path1, path2, style=style, 
                                       connect=False)
                elif data_type == "Curve":
                    self.gb.show_curve(axes, path1, path2, style=style)
                elif data_type == "Segmentation":
                    self.gb.show_pic(axes, path1, path2, gray=True)
                else:
                    self.gb.show_pic(axes, path1, path2)

                # Displays similarity/dissimilarity scores under pair
                scores = ""
                for matrix_name in self.selected_matrices:
                    
                    matrix = self.data_set.matrices[matrix_name]["matrix"]
                    cur_score = str(matrix[self.data_set.records.index(
                        record1)][self.data_set.records.index(record2)])
                    if cur_score == None or cur_score == "nan":
                        cur_score = "Unavailable"
                    scores += (
                        matrix_name + " score: " + cur_score + "\n")
                    
                axes.set_xlabel("\n".join(wrap(scores, 30, 
                                replace_whitespace=False)))
                title_text = "Records %s and %s" % (record1.id, record2.id)

                title_text = "\n".join(wrap(title_text, 30, 
                                       replace_whitespace=False))
                if len(title_text) > 30:
                    axes.set_title(title_text, fontsize=10)
                else:
                    axes.set_title(title_text)

            # For a 1D list, shows a single record.
            else:
                fig.suptitle(title, fontsize=18)
                
                cur_record = records_to_show[index]
                
                cur_path = "Unavailable"
                if data_type in cur_record.files:
                    cur_path = self.data_set.get_file_path(cur_record, 
                                                           data_type)
                    
                if data_type == "Point Cloud":
                    self.gb.show_curve(axes, cur_path, style=style, 
                                       connect=False)
                elif data_type == "Curve":
                    self.gb.show_curve(axes, cur_path, style=style)
                elif data_type == "Segmentation":
                    self.gb.show_pic(axes, cur_path, gray=True)
                else:
                    self.gb.show_pic(axes, cur_path)
                
                axes.set_xlabel("Record %s" % cur_record.id)
        
            axes.get_xaxis().set_ticks([])
            axes.yaxis.set_visible(False) 

        # Adds buttons
        bnext=None
        bprev=None
        
        plots_remaining = show_start + self.settings["Plots per page"][0]
        if plots_remaining < len(self.selected_records):
            bnext = Button(fig.add_axes([0.89, 0.007, 0.1, 0.055]), 'Next')
            bnext.on_clicked(partial(self.change_figure, fig, 1))
            
        if show_start > 0:
            bprev = Button(fig.add_axes([0.78, 0.007, 0.1, 0.055]), 'Previous')
            bprev.on_clicked(partial(self.change_figure, fig, -1))
        
        return bnext, bprev
                
                                                                 
    def change_figure(self, fig, sign=1, event=None):      
        """Changes the figure after pressing the Next or Previous buttons.
        
        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to change.
        sign : {1, -1}
            Determines whether to move forwards (1) or backwards (-1) through 
            the list of records to display.
        event : matplotlib.backend_bases.MouseEvent, optional
            The event that triggered changing the figure.
            
        Returns
        -------
        None
            
        See Also
        --------
        view_selected_records: Sets up initial figure for record viewing.
        add_plots: Adds specific plots inside a figure.
        
        """
                          
        for self.selection_id in self.figure_info:          
            for figure_data in self.figure_info[self.selection_id]:
                if figure_data["figure"] == fig:
                   
                    for all_figure_data in self.figure_info[self.selection_id]:

                        cur_fig = all_figure_data["figure"]
                        cur_fig.clf()
                    
                        # Determines amount of forward/backward change
                        direction_change = self.settings[
                            "Plots per page"][0]*sign   
                                    
                        all_figure_data["start"] += direction_change
                        
                        if all_figure_data["start"] < len(
                            self.selected_records):
                            
                            all_figure_data["buttons"] = self.add_plots(
                                cur_fig, all_figure_data["start"], 
                                all_figure_data["data type"], 
                                all_figure_data["title"])
                            cur_fig.canvas.draw()
                            
                        else:
                            self.gb.loading_error("data files")
                            return
        return
    
    
    def handle_close(self, event):
        """Closes the current figure.
        
        Parameters
        ----------
        event : QEvent, optional
            The event that triggered closing the figure.
            
        Returns
        -------
        None
        """
        self.selected_records=[]
        plt.close()
        gc.collect()
        

    def show_text(self, data_type):
        """Displays a window of text descriptions.
        
        Parameters
        ----------
        data_type : str
            The data type to display.
            
        Returns
        -------
        None
        """
        
        text_widget = self.gb.make_widget(qtw.QStackedWidget(), 
                                          window_title=data_type)
        text_widget.setWindowIcon(self.data_set.icon)
        
        texts_per_pg = self.settings["Plots per page"][0]
        font = qtgui.QFont()

        num_shown = 0
        for records in self.selected_records:
            
            if num_shown % texts_per_pg == 0:
                cur_layer = qtw.QWidget()
                cur_layout = qtw.QGridLayout()
                cur_layer.setLayout(cur_layout)
                text_widget.addWidget(cur_layer)
            
                font.setPointSize(12)
                font.setBold(True)
                
                cur_layout.addWidget(self.gb.make_widget(
                    qtw.QLabel(data_type), font=font), 0, 0)
                
                button_layout = qtw.QHBoxLayout()
                if num_shown > 0:
                    button_layout.addWidget(self.gb.make_widget(
                        qtw.QPushButton("Previous"), "clicked", 
                        partial(self.prev_texts, text_widget)), 
                        2*texts_per_pg+1)
                if num_shown + texts_per_pg < len(self.selected_records):
                    button_layout.addWidget(self.gb.make_widget(
                        qtw.QPushButton("Next"), "clicked", 
                        partial(self.next_texts, text_widget)), 
                        2*texts_per_pg+1)
                cur_layout.addLayout(button_layout, 2*texts_per_pg+3, 0, 1, 2)
            
            font.setPointSize(11)
            font.setBold(False)    

            if isinstance(records, list):
                title_label = "Records "+records[0].id+" and "+records[1].id
                cur_layout.addWidget(self.gb.make_widget(
                    qtw.QLabel(title_label), font=font), 
                    2*(num_shown % texts_per_pg)+1, 0)

                if data_type in records[0].files:
                    path1 = self.data_set.get_file_path(records[0], data_type)
                else:
                    path1 = "Unavailable"   
                    
                edit1 = self.gb.show_text(path1)
                cur_layout.addWidget(edit1, 2*(num_shown % texts_per_pg)+2, 0)
                
                if data_type in records[1].files:
                    path2 = self.data_set.get_file_path(records[1], data_type)
                else:
                    path2 = "Unavailable"
                
                edit2 = self.gb.show_text(path2)
                cur_layout.addWidget(edit2, 2*(num_shown % texts_per_pg)+2, 1)
                
                num_shown+=1
            
            # Shows a 1D record list
            else:
                
                if data_type in records.files:
                    path = self.data_set.get_file_path(records, data_type)
                else:
                    path = "Unavailable"
                    
                edit = self.gb.show_text(path)                
                    
                cur_layout.addWidget(
                    edit, 2*(num_shown % texts_per_pg)+2, 0, 2, 1)
                
                num_shown+=1
        
        self.text_widgets.append(text_widget)            
        text_widget.show()
    
                
    def next_texts(self, widget): 
        """Moves to the next page of text descriptions.
        
        Parameters
        ----------
        widget : QWidget
            The figure to change.
            
        Returns
        -------
        None
        
        """
        
        widget.setCurrentIndex(widget.currentIndex() + 1)
        
    def prev_texts(self, widget):
        """Moves to the previous page of text descriptions.
        
        Parameters
        ----------
        widget : QWidget
            The figure to change.
            
        Returns
        -------
        None
        
        """

        widget.setCurrentIndex(widget.currentIndex() - 1)
        
###############################################################################    
# Selection methods
                      
    def plot_hover(self, plot_type, event):
        """Displays a record's ID when hovering over it on the plot.
        
        Parameters
        ----------
        plot_type : str
            The type of plot being hovered on ("2d", "3d", "cluster")
        event : matplotlib.backend_bases.MouseEvent
            The hover event that triggered displaying the ID.
            
        Returns
        -------
        None
        """
        
        if plot_type == "2d":
            current_axes = self.current_2d_axes
            current_artists = self.artists["2d"]
            current_notes = self.annotations["2d"]
            pt_index = self.pt_index_2d
        elif plot_type == "3d":
            current_axes = self.current_3d_axes
            current_artists = self.artists["3d"]
            current_notes = self.annotations["3d"]
            pt_index = self.pt_index_3d
        else:
            current_axes = self.current_cluster_axes
            current_artists = self.artists["cluster"]
            current_notes = self.annotations["cluster"]
            pt_index = self.pt_index_cluster
        
        # If colored by group
        if self.color_grouping or isinstance(current_artists[0], dict):

            new_index = None
            text = None
            for axis_index, axis in enumerate(current_axes):
                if event.inaxes == axis:
                    
                    for group_name in current_artists[axis_index]:
                        cont, ind = current_artists[axis_index][
                            group_name].contains(event)
                        
                        if cont:
                            new_index = ind["ind"][0]
                            text = current_notes[
                                axis_index][group_name][new_index].get_text()
                            
                            for axis_notes in current_notes:
                                for group in axis_notes:
                                        for note in axis_notes[group]:
                                            if note.get_text() == text:
                                                note.set_visible(True)
                            break
                   
            # Removes old point's annotation
            if pt_index is not None and pt_index != new_index:
                
                for axis_notes in current_notes:
                    for group in axis_notes:
                        for note in axis_notes[group]:
                            
                            cur_text = note.get_text()
                                
                            if cur_text == self.cur_note:
                                note.set_visible(False)
            
            self.cur_note = text
                    
        # If not colored by group            
        else:
            new_index = None
            for axis_index, axis in enumerate(current_axes):
                if event.inaxes == axis:
                  
                    cont, ind = current_artists[axis_index].contains(event)
                    
                    if cont:
                        new_index = ind["ind"][0]
                        self.cur_note=current_notes[0][new_index].get_text()
                        
                        for axis_notes in current_notes:
                            axis_notes[new_index].set_visible(True)
                        break
                  
            # Removes old point's annotation
            if pt_index is not None and pt_index != new_index:
                for axis_notes in current_notes:
                    axis_notes[pt_index].set_visible(False)
        
        if plot_type == "2d":
            self.pt_index_2d = new_index
        else:
            axis.get_figure().canvas.draw()
            
            if plot_type == "3d":
                self.pt_index_3d = new_index
            else:
                self.pt_index_cluster = new_index
        
    
    def plot_pick(self, event):
        """Displays the records associated with a clicked point on a plot.
        
        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The click event that triggered displaying the records.
            
        Returns
        -------
        None
        """
        
        if event.mouseevent.dblclick:
            new_records = [self.data_set.record_from_id(self.cur_note)]
            
            if (len(self.selected_records) == 0 or 
                isinstance(self.selected_records[0], list) or 
                set(tuple(new_records)) != set(tuple(self.selected_records))):
                    
               self.selected_records = new_records
               self.view_selected_records()
               
  
    def plot_rect(self, axis_index=None, eclick=None, erelease=None):
        """Displays all record pairs in the selected rectangle of the plot. 
        
        Parameters
        ----------
        axis_index : int, optional
            The axis of the plot where the selection occurred.
        eclick: matplotlib.backend_bases.MouseEvent, optional
            The event where the user clicked on the plot.
        erelease: matplotlib.backend_bases.MouseEvent, optional
            The event where the mouse was released on the plot.
            
        Returns
        -------
        None
        """

        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        xmax = max(x1, x2)
        xmin = min(x1, x2)
        ymax = max(y1, y2)
        ymin = min(y1, y2)
        
        # Checks that this was not a single click
        if not (x1 == x2 and y1 == y2):

            mtx = self.data_set.matrices[
                self.selected_matrices[axis_index]]["matrix"]
            self.selected_records = []
            
            # Looks for points within the rectangle
            xpts = self.cur_points[axis_index][0]
            ypts = self.cur_points[axis_index][1]
            npts = range(len(xpts))
            pts = [pt for pt in npts 
                   if xmin < xpts[pt] < xmax and ymin < ypts[pt] < ymax]
            for pt_index in pts:
                          
                if self.mv_plot_type  == "roc":
                    
                    # Gets the dissimilarity score for that (fpr, tpr) tuple
                    cur_threshold = self.thresholds[axis_index][pt_index]
                    
                    # Finds the corresponding record pair
                    for row_index, row in enumerate(mtx):
                        for col_index, score in enumerate(row):
                                
                            if round(score, 7) == round(cur_threshold, 7):
                                r1 = self.data_set.records[row_index]
                                r2 = self.data_set.records[col_index]
                                self.selected_records.append([r1, r2])
                else:
                    self.selected_records.append(
                        self.data_set.records[pt_index])
                                              
            self.view_selected_records()
            
   
    def plot_lasso(self, axis_index=None, verts=None):
        """Displays all record pairs in the selected lasso region of the plot.
        
        Parameters
        ----------
        axis_index : int, optional
            The axis of the plot where the selection occurred.
        verts : list of list of float, optional
            The coordinates of the vertices of the lasso area.
            
        Returns
        -------
        None
        """
        
        self.selected_records = [] 
        mtx = self.data_set.matrices[
            self.selected_matrices[axis_index]]["matrix"]
        
        for pt_index, cur_x in enumerate(self.cur_points[axis_index][0]):
            
            cur_y = self.cur_points[axis_index][1][pt_index]
            if path.Path(verts).contains_points([(cur_x, cur_y)]):
                
                if self.mv_plot_type == "roc":
                    
                    # Gets the dissimilarity score for that (fpr, tpr) tuple
                    cur_threshold = self.thresholds[axis_index][pt_index]
                    
                    # Finds the corresponding record pair
                    for row_index, row in enumerate(mtx):
                        for col_index, score in enumerate(row):
                                
                            if round(score, 7) == round(cur_threshold, 7):
                                r1 = self.data_set.records[row_index]
                                r2 = self.data_set.records[col_index]
                                self.selected_records.append([r1, r2])
                else:
                    new_record = self.data_set.records[pt_index]
                    self.selected_records.append(new_record)

        self.view_selected_records()
            

    def mds_selector_change(self, event):
        """Changes MDS selectors from rectangle to lasso or vice versa.
        
        `self.selectors` maintains a reference to each selector so that they 
        can be used after another figure is opened.
        
        Parameters
        ----------
        event: matplotlib.backend_bases.MouseEvent
            The event that triggered the selector change.
            
        Returns
        -------
        None
        """
        
        text = str(self.lasso_rect.label.get_text())
            
        for index, selector in enumerate(self.selectors):

            ax = self.current_2d_axes[index]
            
            if text == "Lasso":
                self.lasso_rect.label.set_text("Rectangle")
                self.selectors[index] = LassoSelector(
                    ax, partial(self.plot_lasso, index))


            else:
                self.lasso_rect.label.set_text("Lasso")
                self.selectors[index] = RectangleSelector(
                    ax, partial(self.plot_rect, index))

    
    def cluster_selector_change(self, event, rect):
        """Changes clustering selectors from rectangle to lasso or vice versa.
        
        `self.selectors` maintains a reference to each selector so that they 
        can be used after another figure is opened.
        
        Parameters
        ----------
        event: matplotlib.backend_bases.MouseEvent
            The event that triggered the selector change.
        rect: {False, True}
            Whether the rectangle seelctor button was pressed.
            
        Returns
        -------
        None
        """
            
        for index, selector in enumerate(self.selectors):

            ax = self.current_cluster_axes[index]
            if rect==True:
                self.rect_btn.setDefault(True)
                self.lasso_btn.setDefault(False)
                self.selectors[index] = RectangleSelector(
                    ax, partial(self.plot_rect, index))

            else:
                self.lasso_btn.setDefault(True)
                self.rect_btn.setDefault(False)
                self.selectors[index] = LassoSelector(
                    ax, partial(self.plot_lasso, index))
     
      
    def heat_click(self, event):
        """Displays the record pair of a clicked point on the heat map.
        
        Parameters
        ----------
        event: matplotlib.backend_bases.MouseEvent
            The click that triggered the record display.
            
        Returns
        -------
        None
        
        See Also
        --------
        view_selected_records: Displays the selected records.
        """
        
        if event.dblclick:
            new_records = [self.data_set.records[int(event.xdata)], 
                           self.data_set.records[int(event.ydata)]]
            is_2d = ((self.selected_records is not None) 
                     and len(self.selected_records) > 0 
                     and isinstance(self.selected_records[0], list))
            if (not is_2d and 
                set(tuple(new_records)) != set(tuple(self.selected_records))):
               self.selected_records = new_records
               self.view_selected_records()
            
    
    def hier_hover(self, event):
        """Displays all child IDs when the user hovers over a node.
        
        Parameters
        ----------
        event: matplotlib.backend_bases.MouseEvent
            The click that triggered the ID display.
            
        Returns
        -------
        None
        
        See Also
        --------
        hierarchical_clustering: Displays the hierarchical clustering.
        """
        
        new_index = None
        new_note = None
        for axis_index, axis in enumerate(self.current_cluster_axes):
            if event.inaxes == axis:
                
                # If the user hovers over a (red) node with several child IDs
                note_index = 2*axis_index
                redcont, redind = self.artists[
                    "cluster"][note_index].contains(event)
                if redcont:
                    
                    new_index = redind["ind"][0]
                    new_note = self.annotations[
                        "cluster"][note_index][new_index].get_text()
                    
                    self.annotations[
                        "cluster"][2*axis_index][new_index].set_visible(True)
                    break
                
                # If the user hovers over a (blue) individual ID
                note_index = 2*axis_index + 1
                bluecont, blueind = self.artists[
                    "cluster"][note_index].contains(event)
                if bluecont:
                    
                    new_index = blueind["ind"][0]
                    new_note = self.annotations[
                        "cluster"][note_index][new_index].get_text()

                    if self.cur_note != new_note:
                        for axis_notes in self.annotations["cluster"][1::2]:
                            for note in axis_notes:
                                if note.get_text() == new_note:
                                    note.set_visible(True)
                    break
                  
        # Removes old point's annotation
        if self.cur_note is not None and self.cur_note != new_note:
            for axis_notes in self.annotations["cluster"]:
                for note in axis_notes:
                    if note.get_text() == self.cur_note:
                        note.set_visible(False)
        
        if self.cur_note != new_note:
            axis.get_figure().canvas.draw()
        
        self.pt_index_cluster = new_index
        self.cur_note = new_note


    def hier_pick(self, event):
        """When the user clicks on a node, displays all its child records.
        
        Parameters
        ----------
        event: matplotlib.backend_bases.MouseEvent
            The click that triggered the record display.
            
        Returns
        -------
        None
        
        See Also
        --------
        hierarchical_clustering: Displays the hierarchical clustering.
        
        """

        if not event.mouseevent.dblclick:
            return
        
        if "IDs" in self.cur_note:            
            ids = str(self.cur_note[4:]).replace("\n", " ").split(",")
            new_records = [self.data_set.record_from_id(cur_id.strip()) 
                           for cur_id in ids]
        else:
            new_records = [self.data_set.record_from_id(str(self.cur_note))]

        if set(tuple(new_records)) != set(tuple(self.selected_records)):
            self.selected_records = new_records
            self.view_selected_records()

    
    def mv_hover(self, event=None):
        """Shows information when hovering over the match visualization plot.
        
        For smooth histograms, displays the heights of the histograms of 
        matched and unmatched scores at that point. For ROC curves, displays 
        the true and false positive rates, as well as the current threshold if 
        hovering over the curve itself. For the unsmoothed histogram and the 
        linear ordering, displays the number of matches and non-matches to 
        either side of the selected point in the table. For the linear 
        ordering, if hovering over a point, also displays the IDs and score of 
        the record pair that the point corresponds to.
        
        Parameters
        ----------
        event: matplotlib.backend_bases.MouseEvent
            The click that triggered the record display.
            
        Returns
        -------
        None
        
        See Also
        --------
        histogram: Displays the histogram
        show_smooth_histogram: Displays the smooth histogram
        show_roc_curve: Displays the ROC curve
        linear_ordering: Displays the linear ordering
        """
        
        new_index = None
        new_notes = []
        new_note = None
        for axis_index, axis_name in enumerate(self.selected_matrices):
            
            
            mnm_scores = self.data_set.match_nonmatch_scores[axis_name]
            match_scores = [row[0] for row in mnm_scores["matches"]]
            nonmatch_scores = [row[0] for row in mnm_scores["nonmatches"]]
            all_scores = [row[0] for row in mnm_scores["all"]] 
            min_score = min(all_scores)
            max_score = max(all_scores)
            
            score = None
            score_text = "Score: " 
        
            # ROC curve: only displays threshold if hovered over the curve 
            if self.mv_plot_type == "roc":
                score_text = "Threshold: "
                
                if (event is not None and event.xdata is not None 
                    and 0 <= event.xdata <= 1 and 0<= event.ydata <= 1):
                    
                    if self.use_1_plot:
                        axis = self.current_axes[0]
                    else:
                        axis = self.current_axes[axis_index]
                    
                    if event.inaxes == axis:
                        cont, ind = self.artists[
                            "mv"][axis_index].contains(event)
                        
                        # Sets note visible and adds to new notes
                        if cont:
                            new_note = self.annotations["mv"][
                                axis_index][ind["ind"][0]]
                            new_note.set_visible(True)
                            score = float(new_note.get_text())
                            score_text += str(format(score, '.4g'))
                            new_notes.append(new_note)
                                
                    score_text += "\nTrue Positive Rate: " + str(
                        format(event.ydata, '.4g'))
                    score_text += "\nFalse Positive Rate: " + str(
                        format(event.xdata, '.4g'))
                    
                else:
                    score_text += "\nTrue Positive Rate: "
                    score_text += "\nFalse Positive Rate: "
                
            # Linear ordering and histograms
            elif (event is not None and event.xdata is not None 
                and event.xdata>=min_score and event.xdata<=max_score):
                
                score = event.xdata
                score_text += str(format(score, '.4g'))
                
                if self.mv_plot_type == "linear ordering" and not new_index:
                    
                    axis = self.current_axes[axis_index]
                    if event.inaxes == axis:
                        
                        cont, ind = self.artists[
                            "mv"][axis_index][0].contains(event)
                        line_index = 0
                        
                        if not cont:
                            cont, ind = self.artists[
                                "mv"][axis_index][1].contains(event)
                            line_index = 1
                        
                        # Sets note visible and adds to new indices
                        if cont:
                            new_index = []
                            cur_note = self.annotations["mv"][
                                axis_index][line_index][ind["ind"][0]]
                            cur_note.set_visible(True)
                            
                            for axis_notes in self.annotations["mv"]:
                                for index, note in enumerate(
                                    axis_notes[line_index]):
                                    
                                    note_text = note.get_text()
                                    cur_text = cur_note.get_text()
                                    
                                    if (note_text[:note_text.index(":")] 
                                        == cur_text[:cur_text.index(":")]):
                                        
                                        note.set_visible(True)
                                        new_index.append(index)

                #Smooth histogram: displays match and nonmatch heights
                elif self.mv_plot_type == "smooth histogram":
                    
                    x = np.linspace(0, max_score, 200)
                    
                    for index, point_set in enumerate(self.cur_points):
                        match_height = np.interp(score, x, point_set[0])
                        nonmatch_height = np.interp(score, x, point_set[1])
                    
                    score_text += "\nMatches' height: " + str(
                        format(match_height, '.6g'))
                    score_text += "\nNon-matches' height: " + str(
                        format(nonmatch_height, '.6g'))
                    score_text += "\nMatch:nonmatch height ratio: " + str(
                        format(match_height/nonmatch_height, '.6g'))

            # Fills in table
            if score:
                arr_matches = np.array(match_scores)
                arr_nonmatches =np.array(nonmatch_scores)
                
                matches_left = np.where(arr_matches <= score)[0].size
                matches_right = np.where(arr_matches > score)[0].size
                nonmatches_left = np.where(arr_nonmatches <= score)[0].size
                nonmatches_right = np.where(arr_nonmatches > score)[0].size
                
                stats = [[matches_left, matches_right, len(arr_matches)], 
                         [nonmatches_left, nonmatches_right, 
                          len(arr_nonmatches)]]
                
                for row in range(2, 4):
                    
                    total = stats[row-2][2]
                    to_add = [str(format(stats[row-2][col]*100.0/total, '.4g'))
                              + "%" for col in range(3)]    
                    stats.append(to_add)
                
                for rownum, row in enumerate(stats):
                    for colnum, item in enumerate(row):
                        self.stats_tables[axis_index].setItem(
                            rownum, colnum, qtw.QTableWidgetItem(str(item)))
            
            # If outside the matrix's range, clear the table
            else:
                if self.mv_plot_type == "smooth histogram":
                    score_text += ("\nMatches' height: \nNon-matches' height:"
                                   "\nRatio:")
                    
                for rownum in range(4):
                    for colnum in range(2):
                        self.stats_tables[axis_index].setItem(
                            rownum, colnum, qtw.QTableWidgetItem(""))
                       
            self.stats_labels[axis_index].setText(score_text)
            
            header = self.stats_tables[axis_index].horizontalHeader()
            header.setStretchLastSection(True)
        
        
        # For ROC and linear ordering, hides old annotations
        if self.mv_plot_type == "linear ordering":
            
            if self.pt_index_mv is not None and self.pt_index_mv != new_index:
                for axis_index, axis_notes in enumerate(
                    self.annotations["mv"]):
                    
                    cur_pt = self.pt_index_mv[axis_index]
                    
                    if len(axis_notes[0]) > cur_pt:
                        axis_notes[0][cur_pt].set_visible(False)
                    if len(axis_notes[1])>cur_pt:
                        axis_notes[1][cur_pt].set_visible(False)
            
            self.pt_index_mv = new_index 
            
        elif self.mv_plot_type == "roc":
            if self.cur_mv_note is not None:
                
                for note in self.cur_mv_note:
                    if note is not None and note not in new_notes:
                        note.set_visible(False)
                        
            self.cur_mv_note = new_notes
        
        if new_note or new_index: 
            self.current_axes[0].get_figure().canvas.draw()
                
     
    def mv_click(self, event=None, prev_score=None):
        """Shows record(s) when the user double-clicks on a match visualization
        
        On the ROC plot, pins lines displaying the true and false positive 
        rates to the plot. On the linear ordering and histograms, draws a 
        vertical line where the user clicked and annotates the score there.
        
        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent, optional
            The click that triggered the record display.
        prev_score : float, optional
            When adding interpolation scores without the user clicking, stores 
            the score to be added.
            
        Returns
        -------
        None
        
        See Also
        --------
        histogram: Displays the histogram
        show_smooth_histogram: Displays the smooth histogram
        show_roc_curve: Displays the ROC curve
        linear_ordering: Displays the linear ordering

        """
        
        if prev_score:
            score = prev_score
        else:
            score = event.xdata
            
        bbox_style = dict(boxstyle="round", fc="w")
        axis = None
        to_store = None
        if prev_score or event.dblclick:
            draw = False
            
            for axis_index, axis_name in enumerate(self.selected_matrices): 
                
                mnm_scores = self.data_set.match_nonmatch_scores[axis_name]
                all_scores = [row[0] for row in mnm_scores["all"]]
                min_score = min(all_scores)
                max_score = max(all_scores)
                
                # If ROC, draw lines and annotate with the TPR and FPR
                if self.mv_plot_type == "roc":
                    if not (self.use_1_plot and axis_index>0):
                    
                        if prev_score or (event.xdata>=0 and event.xdata<=1 
                                          and event.ydata>=0 
                                          and event.ydata<=1):
                            
                            axis = self.current_axes[axis_index]
                            
                            if prev_score:
                                score_x = prev_score[0]
                                score_y = prev_score[1]
                            else:
                                score_x = event.xdata
                                score_y = event.ydata
                                to_store = (score_x, score_y)
                                
                            axis.axvline(score_x, color='k', linestyle='--')
                            axis.axhline(score_y, color='k', linestyle='--')
                            
                            x_note = axis.annotate(
                                s="FPR: " + str(format(score_x, '.3g')), 
                                xy=(score_x, 0), xycoords='data', 
                                xytext=(score_x, -.07), 
                                arrowprops=dict(arrowstyle='->'), 
                                bbox=bbox_style)
                            x_note.draggable()
                            
                            y_note = axis.annotate(
                                s="TPR: " + str(format(score_y, '.3g')), 
                                xy=(0, score_y), xycoords='data', 
                                xytext=(-.16, score_y - .05), 
                                arrowprops=dict(arrowstyle='->'), 
                                bbox=bbox_style)
                            y_note.draggable()
                            
                            draw = True
                
                # If linear ordering, draws vertical line at score
                elif score >= min_score and score <= max_score:
                    
                    axis = self.current_axes[axis_index]
                    axis.axvline(score, color='k', linestyle='--')
                    ymin, ymax = axis.get_ylim()
                    note_height = .75*(ymax+ymin)
                    text_height = (score, note_height)
                    
                    if self.mv_plot_type == "linear ordering":
                        axis.plot(score, .005, 'gs')
                        axis.plot(score, .005, 'gx', markersize=15)
                        axis.plot(score, -.03, 'gs')
                        axis.plot(score, -.03, 'gx', markersize=15)
                        
                        note_height = -.01
                        text_height = (score+max_score/80, -.01)
                        
                    score_note = axis.annotate(
                        s=str(format(score, '.3g')), xy=(score, note_height), 
                        xytext=text_height, arrowprops=dict(arrowstyle='->'), 
                        bbox=bbox_style)
                    score_note.draggable()
                    
                    to_store = score
                    draw = True 
            
            if draw:    
                axis.get_figure().canvas.draw()
                
        if to_store and not prev_score:
            self.interpolation_scores.append(to_store)
                
    
    def mv_pick(self, event):
        """Displays records associated with a clicked point on a plot.
        
        On the linear ordering, displays te record pair associated with the 
        picked point.
        
        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The click that triggered the record display.
            
        Returns
        -------
        None
        
        See Also
        --------
        show_roc_curve: Displays the ROC curve
        linear_ordering: Displays the linear ordering
        
        """
        
        if event.mouseevent.dblclick:
            
            if self.mv_plot_type == "linear ordering":
                
                for axis_index, axis in enumerate(self.current_axes):
                    if event.mouseevent.inaxes == axis:
                        mtx_name = axis.get_title()
                        break
                
                # Check if match or nonmatch picked
                if event.mouseevent.ydata > -.0125:
                    line_type = "matches"
                else:
                    line_type = "nonmatches"
                
                info = self.data_set.match_nonmatch_scores[
                    mtx_name][line_type][self.pt_index_mv[axis_index]]

                record1 = self.data_set.record_from_id(info[1])
                record2 = self.data_set.record_from_id(info[2])
                
                new_records = [record1, record2]
                if (not self.selected_records 
                    or isinstance(self.selected_records[0], list) 
                    or set(tuple(new_records)) 
                    != set(tuple(self.selected_records))):
                    
                   self.selected_records = new_records
                   
                self.view_selected_records()               

    
    def closeEvent(self, event):
        """Closes the window and saves the data set if necessary.
        
        Parameters
        ----------
        event : QEvent
            The event that triggered closing the window.
            
        Returns
        -------
        None
        
        See Also
        --------
        DataSet.save: Saves the data set before closing.
        
        """
        
        # Asks if the user wants to save this set of analyses (i.e. pickle it)
        # Only asks if 1 window is left open
        if not self.data_set.update and self.data_set.num_widgets_open < 2:
                close = self.data_set.save(self)
                if close:
                    plt.close("all")
                    app = qtw.QApplication.instance()
                    app.closeAllWindows()
                    event.accept()
                    
                else:
                    event.ignore()
                    return
        
        self.data_set.num_widgets_open -= 1

      
    class CustomCursor(MultiCursor):
        """A cursor with two custom horizontal lines and a vertical line.
        
        Used in the smooth histogram to display the heights of the histograms 
        of the matched and unmatched scores at the same time.        

        Parameters
        ----------
        canvas : matplotlib.backend_bases.FigureCanvas
            The canvas on which the cursor is displayed
        axes : matplotlib.axes.Axes
            The axes on which the cursor is displayed
        useblit : {True, False}, optional
            Determines whether to use blitting when rendering the cursor.
        horizOn : {False, True}, optional
            Whether to display a horizontal line
        vertOn : {True, False}, optional
            Whether to display a vertical line
        points : list of list, optional
            The list of points on display.
        visible : {True, False}, optional
            Determines whether to display the cursor
        mnm : dict of {str : dict of {str : list}}
            For each matrix, stores lists of match scores, nonmatch scores,
            and all scores; same format as DataSet.match_nonmatch_scores.
        selected_matrices : list of str, optional
            The names of the matrices on display.
        
        Other Parameters
        ----------------
        linewidth : int
            The width of the cursor lines.
        ls : str
            The matplotlib line style of the cursor lines.
        color : str
            The cursor color.
        
        Attributes
        ----------
        match_nonmatch_scores : dict of list
            For each matrix, stores lists of match scores, nonmatch scores,
            and all scores; same format as DataSet.match_nonmatch_scores.
        cur_points : list of list
            The list of points being displayed.
        canvas : matplotlib.backend_bases.FigureCanvas
            The canvas on which the cursor is displayed
        axes : matplotlib.axes.Axes
            The axes on which the cursor is displayed
        horizOn : {False, True}, optional
            Whether to display a horizontal line
        vertOn : {True, False}, optional
            Whether to display a vertical line
        visible : {True, False}, optional
            Determines whether to display the cursor
        useblit : {True, False}, optional
            Determines whether to use blitting when rendering the cursor.
        
        """
                     
        def __init__(self, canvas, axes, useblit=True, horizOn=False, 
                     vertOn=True, points=None, mnm=None, 
                     selected_matrices=None, **lineprops):
            

            self.match_nonmatch_scores = mnm
            self.selected_matrices = selected_matrices
            self.cur_points = points
            
            self.canvas = canvas
            self.axes = axes
            self.horizOn = horizOn
            self.vertOn = vertOn
    
            xmin, xmax = axes[-1].get_xlim()
            ymin, ymax = axes[-1].get_ylim()
            xmid = 0.5 * (xmin + xmax)
            ymid = 0.5 * (ymin + ymax)
    
            self.visible = True
            self.useblit = useblit and self.canvas.supports_blit
            self.background = None
            self.needclear = False
    
            if self.useblit:
                lineprops['animated'] = True
    
            if vertOn:
                self.vlines = [ax.axvline(xmid, visible=False, **lineprops)
                               for ax in axes]
            else:
                self.vlines = []
    
            if horizOn:
                self.hlines = [ax.axhline(ymid, visible=False, **lineprops)
                               for ax in axes]
                self.hlines += [ax.axhline(ymid, visible=False, **lineprops)
                               for ax in axes]
            else:
                self.hlines = []
    
            self.connect()      
            
            
        def onmove(self, event):
            """Updates the cursor's lines when moving the mouse.
            
            Parameters
            ----------
            event : matplotlib.backend_bases.MouseEvent
                The event that triggered changing the lines
                
            Returns
            -------
            None
            
            See Also
            --------
            smooth_histogram: Dispalys the smooth histogram that uses this 
                cursor.
            
            """

            if event.inaxes is None:
                return
            if not self.canvas.widgetlock.available(self):
                return
            self.needclear = True
            if not self.visible:
                return
                
            # Finds the heights of the histograms at the current x-coordinate
            score = event.xdata
            heights = []
            for index, matrix_name in enumerate(self.selected_matrices):
                
                mnm_scores = self.match_nonmatch_scores[matrix_name]
                max_score = max([row[0] for row in mnm_scores["all"]])  
                x = np.linspace(0, max_score, 200)
                heights.append(
                    [np.interp(score, x, self.cur_points[index][0]), 
                     np.interp(score, x, self.cur_points[index][1])])
            
            # Redraws the vertical lines at the current x-coordinate
            if self.vertOn:
                for line in self.vlines:
                    line.set_xdata((event.xdata, event.xdata))
                    line.set_visible(self.visible)
                    
            # Redraws the horizontal lines at the current y-coordinate
            if self.horizOn:
                slice_end = int(len(self.hlines)/2)
                for index, line in enumerate(self.hlines[:slice_end]):
                    
                    line.set_ydata((heights[index][0], heights[index][0]))
                    line.set_visible(self.visible)
                    line.set_color("blue")
                    
                    self.hlines[index + slice_end].set_ydata((
                        heights[index][1], heights[index][1]))
                    self.hlines[index + slice_end].set_visible(self.visible)
                    self.hlines[index + slice_end].set_color("red")

            self._update() 
            
        
        def _update(self):
            """Redraws the cursor's lines.
            
            `useblit` determines whether to use blitting when redrawing. 
            Restores the background if necessary.
            
            Parameters
            ----------
            None
                
            Returns
            -------
            None
            """
            
            if self.useblit:
                if self.background is not None:
                    self.canvas.restore_region(self.background)
                if self.vertOn:
                    for ax, line in zip(self.axes, self.vlines):
                        ax.draw_artist(line)
                if self.horizOn:
                    for index, ax in enumerate(self.axes):
                        ax.draw_artist(self.hlines[index])
                        ax.draw_artist(
                            self.hlines[index + int(len(self.hlines)/2)])
                        
                self.canvas.blit(self.canvas.figure.bbox)
            else:
                self.canvas.draw_idle()
                