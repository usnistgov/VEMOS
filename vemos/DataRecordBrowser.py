# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:01:21 2018

@author: Eve Fleisig

This module allows the user to examine and edit individual data records 
associated with images, segmentations, and curves.
"""

import numpy as np
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import os
from functools import partial
from collections import OrderedDict

import PyQt5.QtGui as qtgui
import PyQt5.QtWidgets as qtw
import PyQt5.QtCore as qtcore
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas)
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar)

import vemos.GUIBackend as guibackend
import vemos.VisualMetricAnalyzer as vma


class DataRecordBrowser(qtw.QMainWindow):
    """The main GUI for examining individual records in a data set.

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
    
    browse_index : int
        The index of the record on display.
    cur_match_id : str
        The ID of the currently selected match.
    cur_group : str
        The currently selected group.
        
    cur_figure : matplotlib.backend_bases.FigureCanvas
        The current figure on which images are being displayed.
    cur_axes : matplotlib.axes.Axes
        The current set of axes on which images are being displayed.
    canvas : matplotlib.backend_bases.FigureCanvas
        The canvas of the current figure.
    gb : GUIBackend
        Instance of GUIBackend for display methods.
    """
    
    def __init__(self, data_set):
        
        super(DataRecordBrowser, self).__init__()
        app = qtw.QApplication.instance()                  # For screenshots
        f = qtgui.QFont()
        f.setPointSize(10)
        app.setFont(f)

        self.data_set = data_set
        self.gb = guibackend.GUIBackend()
        
        # Adds menu bar
        menu_bar = self.menuBar()
        menu_bar.clear()
        file_menu = menu_bar.addMenu("File")
        
        file_labels = ['Update loaded data', 'Save analyses', 
                       'Open data description file', 
                       'Open Visual Metric Analyzer', 'Save data description', 
                       'Save data description as...']        
        file_methods = [self.update_data, 
                        partial(self.data_set.save, self, False), 
                        self.open_description, self.open_analyzer, 
                        self.save_description, self.save_description_as]
                     
        for index, label in enumerate(file_labels):
            file_menu.addAction(self.gb.make_widget(
                qtw.QAction(label, self), "triggered", file_methods[index]))
        
        menu_bar.addAction(self.gb.make_widget(qtw.QAction(
            'Generate Matrix', self), "triggered", self.make_matrices))   
        
        # Sets up display        
        self.setGeometry(40, 70, 1100, 900)
        self.setWindowIcon(self.data_set.icon)
        self.create_main_frame()
        self.show_record()
        self.show()
        
        self.data_set.num_widgets_open += 1
        
        
###############################################################################
# Display methods
        
    # Creates the main frame of the GUI .    
    def create_main_frame(self):
        """ Creates the main frame of the GUI.
        
        Creates all figures, labels, and widgets, the toolbar, and the menu. 
        Unlike __init__, this method contains all information that might change 
        after calling update_data.
        
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
        
        self.browse_index = 0
        self.cur_match_id = None
        self.cur_group = None
        self.cur_axes = None
        self.cur_figure = None
        self.canvas = None
        
        
        self.title = 'Data Record Browser: ' + self.data_set.name
        self.setWindowTitle(self.title)
        
        self.main_frame = qtw.QWidget()
        
        self.cur_figure = mpl.figure.Figure()
        self.canvas = FigureCanvas(self.cur_figure)
        self.canvas.setParent(self.main_frame)
        self.canvas.setFocusPolicy(qtcore.Qt.StrongFocus)
        self.canvas.setFocus()
        
        mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)

        # Adds widget for options below figures
        self.feature_widget = qtw.QWidget(self)
        
        # Creates all buttons
        self.buttons = {}
        buttontexts = ["First", " Jump Backward ", "Previous", "Next", 
                       " Jump Forward ", "Last", "Go to",  "Add to new group", 
                       "Remove from current group", "View", "Add match", 
                       "Remove match", "Preview", "Preview all"]
        
        buttonmethods = [partial(self.show_record, 0), 
                         partial(self.move_x_records, None, "Back", -1), 
                         partial(self.move_x_records, -1), 
                         partial(self.move_x_records, 1), 
                         partial(self.move_x_records, None, "Forward", 1),
                         partial(self.show_record, 
                         len(self.data_set.records)-1), self.go_to_record, 
                         self.add_group, self.remove_group, self.view_match, 
                         self.add_match, self.remove_match, self.preview_match,
                         self.preview_all_matches]
                         
        for index, cur_text in enumerate(buttontexts):
            curmethod = buttonmethods[index]
            self.buttons[cur_text] = self.gb.make_widget(
                qtw.QPushButton(cur_text, self.feature_widget), "clicked", 
                curmethod)
        
        label_grid_box = qtw.QGridLayout()
        
        # Adds buttons for changing record on display
        buttons_layout = qtw.QHBoxLayout()
        buttons_layout.addStretch(3)
        for text in buttontexts[:7]:
            buttons_layout.addWidget(self.buttons[text])
            buttons_layout.addStretch(1)
        buttons_layout.addStretch(2)
        
        # Adds file information
        files_scroll = qtw.QScrollArea() 
        files_scroll.setWidgetResizable(True)        
        files_scroll_container = qtw.QWidget()
        files_scroll.setWidget(files_scroll_container)

        self.files_box = self.gb.make_widget(
            qtw.QGroupBox("Files", self.feature_widget))
        self.files_box.setLayout(qtw.QVBoxLayout())
        self.files_box.layout().addWidget(files_scroll)
        
        self.files_box.setMinimumHeight(100)
        self.files_layout = qtw.QVBoxLayout(files_scroll_container)
        
        # Adds group information
        groups_layout = qtw.QHBoxLayout()
        groups_layout.addWidget(qtw.QLabel("Groups: ", self.feature_widget))
        self.group_view = self.gb.make_widget(qtw.QListWidget(
            self.feature_widget), "itemClicked", self.group_chosen)
        groups_layout.addWidget(self.group_view)
        groups_layout.addWidget(self.buttons["Add to new group"])
        groups_layout.addWidget(self.buttons["Remove from current group"])
        groups_box = qtw.QGroupBox("Groups", self.feature_widget)
        groups_box.setLayout(groups_layout)
        groups_box.setMaximumHeight(75)
            
        # Adds match information
        matches_layout = qtw.QGridLayout()
        matches_label = qtw.QLabel("Match IDs: ", self.feature_widget)
        matches_layout.addWidget(matches_label, 0, 0)

        self.match_view = self.gb.make_widget(qtw.QListWidget(
            self.feature_widget), "itemClicked", self.match_chosen)
        matches_layout.addWidget(self.match_view, 0, 1, 2, 1)
        matches_layout.addWidget(self.buttons["Preview all"], 0, 2)
        matches_layout.addWidget(self.buttons["Preview"], 0, 3)
        matches_layout.addWidget(self.buttons["View"], 0, 4)
        matches_layout.addWidget(self.buttons["Add match"], 1, 2)
        matches_layout.addWidget(self.buttons["Remove match"], 1, 3)
        
        matches_box = qtw.QGroupBox("Matches", self.feature_widget)
        matches_box.setLayout(matches_layout)
        #matches_box.setMinimumHeight(80)
                                             
        # Adds box for text description, if necessary
        self.text_edits = []
        show_text = False
        self.text_box = qtw.QGroupBox("Text Description", self.feature_widget)
        for dtype in self.data_set.data_types:
            if self.data_set.data_types[dtype]["type"]=="Description":
                
                if not show_text:
                    text_layout = qtw.QHBoxLayout()                       
                    text_layout.setAlignment(qtcore.Qt.AlignTop)
                    text_layout.addSpacing(5)
                    self.text_box.setLayout(text_layout)
                    show_text = True
                
                text_scroll = qtw.QScrollArea() 
                text_scroll.setWidgetResizable(True)        
                text_scroll_container = qtw.QWidget()
                text_scroll.setWidget(text_scroll_container)
                text_edit = qtw.QTextEdit()
                text_edit.setReadOnly(True)
                self.text_edits.append(text_edit)
                
                text_layout.addWidget(text_edit)

        if not show_text:
            self.text_box.setEnabled(False)
            self.text_box.setVisible(False)
        
        self.text_box.setMaximumHeight(80)
                           
        # Adds information related to score comparisons    
        self.scores_box = qtw.QGroupBox("Scores", self.feature_widget)
        
        if self.data_set.matrices != {}:
            scores_scroll = qtw.QScrollArea() 
            scores_scroll.setWidgetResizable(True)        
            scores_scroll_container = qtw.QWidget()
            scores_scroll.setWidget(scores_scroll_container)
            
            self.scores_table = qtw.QTableWidget(scores_scroll_container)
            self.scores_table.setEditTriggers(
                qtw.QAbstractItemView.NoEditTriggers)

            self.scores_table.setColumnCount(len(self.data_set.matrices))
            self.scores_table.setRowCount(6)
            self.scores_table.setVerticalHeaderLabels([
                "Matrix", "Type", "Most Similar Record", "Score", 
                "Least Similar Record", "Score"])                          
            
            self.scores_table.verticalHeader().setDefaultSectionSize(33)
            
            scores_layout = qtw.QVBoxLayout()                       
            scores_layout.setAlignment(qtcore.Qt.AlignTop)
            scores_layout.addWidget(self.scores_table)
            scores_layout.addSpacing(5)
            self.scores_box.setLayout(scores_layout)
    
        # Disables score-related informations if scores were not loaded
        if not self.data_set.matrices:
            self.scores_box.setEnabled(False)               
        
        label_grid_box.addLayout(buttons_layout, 0, 0, 1, 2)
        label_grid_box.addWidget(self.files_box, 1, 0, 1, 1)
        label_grid_box.addWidget(groups_box, 2, 0, 1, 1)
        label_grid_box.addWidget(matches_box, 3, 0, 1, 1)
        label_grid_box.addWidget(self.text_box, 4, 0, 1, 2)
        label_grid_box.addWidget(self.scores_box, 1, 1, 3, 1)
        label_grid_box.setColumnStretch(1, 2)
        self.feature_widget.setLayout(label_grid_box)
        self.feature_widget.setFixedHeight(400)
                   
        vbox = qtw.QVBoxLayout()
        vbox.addWidget(self.canvas)
        vbox.addWidget(mpl_toolbar)
        vbox.addWidget(self.feature_widget)
                
        self.main_frame.setLayout(vbox)
        self.setCentralWidget(self.main_frame)
        
        
###############################################################################
    def show_record(self, new_index=None):
        """ Displays the data types for the current record on the figure.
        
        Displays the images on the figure, then displays the file names, lists 
        of matches and groups, and score-related information.
        
        Parameters
        ----------
        new_index : int, optional
            The index of the new record to display.
            
        Returns
        -------
        None
        
        See Also
        --------
        create_main_frame: Creates the main frame of the GUI.
        
        """
        
        # Updates current record index if necessary
        if new_index is not None and self.is_valid_index(new_index):
            self.browse_index = new_index
        
        # Removes old file names below figure
        for layout_index in reversed(range(len(self.files_layout))):
            
            cur_layout = self.files_layout.itemAt(layout_index)
            
            for item_index in reversed(range(len(cur_layout))):
                cur_layout.itemAt(item_index).widget().close()
            
            self.files_layout.removeItem(cur_layout)
            cur_layout.setParent(None)
        
        # Displays images on figure
        cur_record = self.data_set.records[self.browse_index]
        cur_files = OrderedDict()
        num_data_types = len(cur_record.files)
        
        # Determines whether to display text descriptions
        #show_text = False
        for dtype in self.data_set.data_types:
            if dtype in cur_record.files:
                if self.data_set.data_types[dtype]["type"]=="Description":
                    num_data_types -= 1
                    #show_text = True
                #else:
                cur_files[dtype] = cur_record.files[dtype]
        
        
        if self.cur_axes:
            for axis in self.cur_axes:
                self.cur_figure.delaxes(axis)  
                
        self.cur_axes = [None for axis in range(num_data_types)]
        
        axis_index = 0
        for index, data_type_name in enumerate(cur_files.keys()):
            
            cur_path = self.data_set.get_file_path(cur_record, data_type_name)
            ftype = self.data_set.data_types[data_type_name]["type"]
            
            # Displays text descriptions at the bottom
            if ftype=="Description":
                self.text_edits[index-axis_index].setText("")
                try:
                    with open(cur_path, "r") as text_file:
                        text = text_file.read()
                        self.text_edits[index-axis_index].setText(text)
                except (IOError, OSError, UnicodeDecodeError, 
                        SyntaxError) as e:
                    self.gb.loading_error("text description", v2=True)
            
            # Displays non-text files on the figure
            else:          
                self.cur_axes[axis_index] = self.cur_figure.add_subplot(
                    1, num_data_types, axis_index + 1)

                try:
                    if ftype=="Curve":
                        self.gb.show_curve(self.cur_axes[axis_index], cur_path, 
                                           None, True)
                    elif ftype=="Point Cloud":
                        self.gb.show_curve(self.cur_axes[axis_index], cur_path, 
                                           None, True, connect=False)
                    elif ftype=="Image":
                        self.gb.show_pic(self.cur_axes[axis_index], cur_path, 
                                         None, True)
                    else:
                        self.gb.show_pic(self.cur_axes[axis_index], cur_path, 
                                         None, True, True)
                        
                except (IOError, ValueError):
                    self.gb.general_msgbox("Loading Failed", (
                        "Unable to load files. Please check that the data set "
                        "was loaded correctly."))
                    self.data_set.clear_data()
                    self.update_data()
                    return
               
                self.cur_axes[axis_index].set_aspect('equal', 'box')
                self.cur_axes[axis_index].get_xaxis().set_ticks([])
                self.cur_axes[axis_index].get_yaxis().set_visible(False)
                self.cur_axes[axis_index].set_xlabel(data_type_name)
                axis_index += 1

            # Adds new file names below figure
            cur_file_layout = qtw.QHBoxLayout()
            cur_label = qtw.QLabel(data_type_name + " file:")
            cur_edit = qtw.QLineEdit(cur_path)
            cur_edit.setReadOnly(True)
            cur_file_layout.addWidget(cur_label)
            cur_file_layout.addWidget(cur_edit)
            self.files_layout.addLayout(cur_file_layout)
            
        self.cur_figure.suptitle("Record " + cur_record.id)
        self.canvas.draw()


        # Displays list of groups for the current record
        self.group_view.clear()
        for group in cur_record.groups:
            self.group_view.addItem(group)
        
        # Displays list of matches for the current record
        self.match_view.clear()
        for match in cur_record.matches:
            self.match_view.addItem(match)
                         
          
        # Displays score-related information, if applicable
        if len(self.data_set.matrices) > 0:
            self.scores_table.setColumnCount(len(self.data_set.matrices))
            for matrix_num, matrix_name in enumerate(self.data_set.matrices):
                cur_matrix = self.data_set.matrices[matrix_name]
                row_array = np.delete(
                    np.array(cur_matrix["matrix"][self.browse_index]), 
                    self.browse_index)
                
                # Adds scores
                try:
                    max_col = np.nanargmax(row_array)
                    max_value = row_array[max_col]
                    max_id = self.data_set.records[max_col].id
                    min_col = np.nanargmin(row_array)
                    min_value = row_array[min_col]
                    min_id = self.data_set.records[min_col].id
                
                
                    if cur_matrix["type"]=="Distance/Dissimilarity":
                        temp = max_value
                        max_value = min_value
                        min_value = temp
                        
                    max_button = qtw.QPushButton(max_id)
                    max_button.clicked.connect(partial(
                        self.show_record, self.data_set.index_from_id(max_id)))
                    min_button = qtw.QPushButton(min_id)
                    min_button.clicked.connect(partial(
                        self.show_record, self.data_set.index_from_id(min_id)))
                
                    score_items = [matrix_name, cur_matrix["type"], 
                                   "", str(max_value), "", str(min_value)]
                    for index, item in enumerate(score_items):
                        self.scores_table.setItem(index, matrix_num, 
                                                  qtw.QTableWidgetItem(item))
                    
                    self.scores_table.setCellWidget(2, matrix_num, max_button)
                    self.scores_table.setCellWidget(4, matrix_num, min_button)
                
                except ValueError:
                    score_items = [matrix_name, cur_matrix["type"], 
                                   "N/A", "N/A", "N/A", "N/A"]
                    for index, item in enumerate(score_items):
                        self.scores_table.setItem(index, matrix_num, 
                                                  qtw.QTableWidgetItem(item))
    
            
            self.scores_table.resizeColumnsToContents()
            
            if len(self.data_set.matrices) < 3:
                self.scores_table.horizontalHeader().setSectionResizeMode(
                    matrix_num, qtw.QHeaderView.Stretch)
                
        # Disables buttons
        self.buttons["Remove from current group"].setEnabled(False)
        self.buttons["Preview"].setEnabled(False)
        self.buttons["View"].setEnabled(False)
        self.buttons["Remove match"].setEnabled(False)


###############################################################################
# File menu methods

    def open_description(self):
        """Opens the description file as a text file. 
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        See Also
        --------
        DataSet.open_description_file: opens the description file
        
        """
        
        if self.windowTitle()[-1]=="*":
            self.save_as()
        
        self.data_set.open_description_file()
        
    
    def save_description(self):
        """ Saves the data description in the current working file.   
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        See Also
        --------
        DataSet.save_current_description_file : Saves the description file.
        
        """
        
        if self.data_set.save_current_description_file():
            self.setWindowTitle(self.title)
    
    
    def save_description_as(self):
        """ Saves the current data description in the file the user chooses.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        See Also
        --------
        DataSet.save_description_file_as : Saves the description file in the 
            location of the user's choice.
            
        """
        
        if self.data_set.save_description_file_as():
            self.setWindowTitle(self.title)
        
    
    def open_analyzer(self):
        """Opens an instance of the Visual Metric Analyzer.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        """
        
        # Prompts user to load a similarity/dissimilarity matrix if needed
        if len(self.data_set.matrices)==0:
            self.gb.general_msgbox(
                "Add Score Matrix", ("Please load a similarity/dissimilarity "
                "matrix to use the Visual Metric Analyzer."))
            self.update_data()
        else:
            self.data_set.update = True
            vma.VisualMetricAnalyzer(self.data_set)
            self.data_set.update = False
    
    def update_data(self):
        """Allows the user to update the data currently loaded data. 
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        See Also
        --------
        DataSet.create_data_loading_widget: opens the widget for updating data.
        """
        self.data_set.update = True
        self.data_set.create_data_loading_widget()
        
        if self.data_set.update:
            self.close()
            self.data_set.update = False
            
            if self.data_set.interface_to_open == "Data Record Browser":
                
                # Sets up display
                self.create_main_frame()
                self.show_record()
                self.data_set.num_widgets_open += 1
                self.show()
            else:
                vma.VisualMetricAnalyzer(self.data_set)

    
###############################################################################
# Button methods
    
    def match_chosen(self, item):
        """Updates the current chosen match.
        
        Enables buttons that require selecting a match.
        
        Parameters
        ----------
        item : QComboBoxItem
            The match item chosen.
            
        Returns
        -------
        None
        
        """
        
        self.cur_match_id = str(item.text())
        self.buttons["Preview"].setEnabled(True)
        self.buttons["View"].setEnabled(True)
        self.buttons["Remove match"].setEnabled(True)

    
    def group_chosen(self, item):
        """Updates the current chosen group.
        
        Enables buttons that require selecting a group.
        
        Parameters
        ----------
        item : QComboBoxItem
            The group item chosen.
            
        Returns
        -------
        None
        
        """
        
        self.cur_group = str(item.text())
        if self.cur_group in self.data_set.groupings["Manual"]:
            self.buttons["Remove from current group"].setEnabled(True)
   
   
    def move_x_records(self, distance=None, direction=None, sign=1):
        """Moves forward or backward by the given number of records.
        
        If `direction` is passed in, asks the user how far to move. Otherise, 
        moves 1 record forward or backward.        
        
        Parameters
        ----------
        distance : int, optional
            The number of records to move forward or backward.
        direction : str, optional
            The direction in which to move ("forward" or "backward")
        sign : {1, -1}, optional
            The sign of the direction in which to move (1=forward, 2=backward)
            
        Returns
        -------
        None
        
        See Also
        --------
        show_record : Displays the record at the parameter index. 
        
        """
        
        if direction:
            move_question = 'Go ' + direction.lower() + ' __ spots:'
            distance, ok = qtw.QInputDialog.getText(self, 'Jump '+ direction, 
                                                    move_question)
            distance = str(distance)       
            
            if not ok:
                return
            if not (distance.isdigit() and self.is_valid_index(int(distance))):
                return self.move_x_records(None, direction, sign)
                
        self.show_record(self.browse_index + sign*int(distance))


    def go_to_record(self):
        """Requests an ID from the user and displays the record with that ID.
               
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        
        See Also
        --------
        show_record : Displays the record at the parameter index. 
        """
        
        # Requests the ID
        ids = [record.id for record in self.data_set.records]        
        cur_id, ok = qtw.QInputDialog.getItem(self, 'Go To...', 'Record ID:',
                                              ids)

        if ok:            
            self.show_record(self.data_set.index_from_id(str(cur_id)))
    
    
    def add_group(self):
        """Adds the current record to a new group.
               
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        
        """
        
        all_groups = []
        for grouping in self.data_set.groupings:
            all_groups += [k for k in self.data_set.groupings[grouping].keys()]
            
        group_name, ok = qtw.QInputDialog.getItem(
            self, 'Add to New Group:', 'New group', 
            all_groups)
        group_name = str(group_name)
        
        cur_record = self.data_set.records[self.browse_index]
        
        # Updates list of groups and record information
        if ok and not group_name in cur_record.groups:
            cur_id = cur_record.id
            
            if group_name in self.data_set.groupings["Manual"]:
                self.data_set.groupings["Manual"][group_name].append(cur_id)                
            else:
                self.data_set.groupings["Manual"][group_name] = [cur_id]              
                
            self.data_set.records[self.browse_index].groups.append(group_name)
            self.group_view.addItem(group_name)    
            self.setWindowTitle(self.title + "*")

    
    def remove_group(self):
        """Removes the current record from the chosen group.
               
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        
        """
        
        cur_record = self.data_set.records[self.browse_index]
        
        if self.cur_group in cur_record.groups:
            cur_record.groups.remove(self.cur_group)
            if len(self.data_set.groupings["Manual"][self.cur_group]) == 1:
                del self.data_set.groupings["Manual"][self.cur_group]
            else:
                self.data_set.groupings[
                    "Manual"][self.cur_group].remove(cur_record.id)
            
            self.group_view.takeItem(self.group_view.currentRow())
            self.setWindowTitle(self.title + '*')
            
            self.cur_group = None
            self.buttons["Remove from current group"].setEnabled(False)
    
    
    def view_match(self):
        """Displays the chosen match.
               
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        
        See Also
        --------
        show_record : Displays the record at the parameter index. 
        
        """
        
        self.show_record(self.data_set.index_from_id(self.cur_match_id))
        
   
    def add_match(self):
        """Adds a new ID to the current record's list of matches. 
        
        Requests the ID from the user in a dialog.
               
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        
        """
        
        cur_record = self.data_set.records[self.browse_index]
        all_records = [record.id for record in self.data_set.records]
        match_id, ok = qtw.QInputDialog.getItem(self, 'Add Match', 
                                                  'Match ID:', all_records)
        match_id = str(match_id)
        match_record = self.data_set.record_from_id(match_id)
        
        if ok:
            if match_record == -1:
                self.gb.general_msgbox("Record Not Found", 
                                       "Please enter a valid record ID.")
                return
                
            if match_id not in cur_record.matches:
                cur_record.matches.append(match_id)
                match_record.matches.append(cur_record.id)
                self.match_view.addItem(match_id)
                #self.setWindowTitle(self.title + '*')

  
    def remove_match(self):
        """Removes the chosen ID from the current record's list of matches. 
        
        Requests the ID from the user in a dialog.
               
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        
        """
        
        cur_record = self.data_set.records[self.browse_index]
        
        if self.cur_match_id in cur_record.matches:
            cur_record.matches.remove(self.cur_match_id)
    
            cur_match = self.data_set.record_from_id(self.cur_match_id)
            cur_match.matches.remove(cur_record.id)
            
            self.match_view.takeItem(self.match_view.currentRow())
            self.setWindowTitle(self.title + '*')
            
            self.cur_match_id = None
            self.buttons["Preview"].setEnabled(False)
            self.buttons["View"].setEnabled(False)
            self.buttons["Remove match"].setEnabled(False)
    
    
    def preview_match(self):
        """Displays the chosen match in a new window. 
               
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        
        See Also
        --------
        GUIBackend.show_curve: Displays a curve file on the plot.
        GUIBackend.show_image: Displays an image file on the plot.
        
        """

        match_record = self.data_set.record_from_id(self.cur_match_id)
        match_files = match_record.files
        
        fig = plt.figure()
        plt.get_current_fig_manager().window.setGeometry(800, 300, 600, 400)
        fig.suptitle("ID " + self.cur_match_id, fontsize=18)
        

        non_text_types = [key for key in match_files.keys() 
                          if key != "Description"]
        for type_index, data_type in enumerate(non_text_types):
                      
            cur_axis = fig.add_subplot(1, len(non_text_types), type_index + 1) 
            cur_path = os.path.join(self.data_set.directory_path, 
                                    match_files[data_type])
            
            if data_type == "Point Cloud":
                self.gb.show_curve(cur_axis, cur_path, connect=False)
            elif data_type == "Curve":
                self.gb.show_curve(cur_axis, cur_path)
            else:
                self.gb.show_pic(cur_axis, cur_path)

            cur_axis.get_xaxis().set_ticks([])
            cur_axis.get_yaxis().set_visible(False)
            cur_axis.set_xlabel(data_type)

        fig.show()
    
    
    def preview_all_matches(self):
        """Displays all matches of the current record in a new window.
               
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        
        See Also
        --------
        GUIBackend.show_curve: Displays a curve file on the plot.
        GUIBackend.show_image: Displays an image file on the plot.
        """
        
        cur_record = self.data_set.records[self.browse_index]
        cur_matches = cur_record.matches

        fig = plt.figure()
        fig.suptitle("Matches for Record " + cur_record.id, fontsize=18)        
        fig.set_tight_layout({"pad": 2.5})   
        
        for match_num, match_id in enumerate(cur_matches):
        
            match_record = self.data_set.record_from_id(match_id)
            match_files = match_record.files
            
            non_text_types = [key for key in match_files.keys() 
                          if key != "Description"]
            for type_index, data_type_name in enumerate(non_text_types):
                
                ftype = self.data_set.data_types[data_type_name]["type"]

                subplot_index = match_num*len(non_text_types) + type_index + 1            
                cur_axis = fig.add_subplot(len(cur_matches), 
                                           len(non_text_types), 
                                           subplot_index) 
                cur_path = os.path.join(self.data_set.directory_path, 
                                        match_files[data_type_name])
                
                if ftype == "Point Cloud":
                    self.gb.show_curve(cur_axis, cur_path, connect=False)
                elif ftype == "Curve":
                    self.gb.show_curve(cur_axis, cur_path)
                else:
                    self.gb.show_pic(cur_axis, cur_path)
                    
                cur_axis.get_xaxis().set_ticks([])
                cur_axis.get_yaxis().set_visible(False)
                cur_axis.set_xlabel(match_id + " " + data_type_name)
                subplot_index+=1
                
            fig.show()

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
            self.show_record()
            self.data_set.update = False
    
###############################################################################
    def is_valid_index(self, index):
        """Returns whether the parameter is a valid record index.
        
        Record indices must be between 0 and the total number of records.
               
        Parameters
        ----------
        index : int
            The index to check.
            
        Returns
        -------
        None
        
        """
        
        return index is not None and 0 <= index < len(self.data_set.records)
    

###############################################################################    
    def closeEvent(self, event):
        """Asks about saving before closing the GUI.
        
        Parameters
        ----------
        event : QEvent, optional
            The event that triggered closing the figure.
            
        Returns
        -------
        None
        
        """
        
        if not self.data_set.update and self.data_set.num_widgets_open < 2:        
            close = self.data_set.save(self)
            if close:
                app = qtw.QApplication.instance()
                app.closeAllWindows()
                plt.close("all")
                event.accept()
            else:
                event.ignore()
                return
        
        self.data_set.num_widgets_open -= 1
    