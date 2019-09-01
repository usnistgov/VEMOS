# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 16:59:17 2017

@author: Eve Fleisig


This class represents a set of data records, associated groupings, and 
dissimilarity matrices.
    
"""

import gc
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Qt5Agg')
import random                                                           #del later
import os
import sys
import csv
import pickle
import re as re
from functools import partial
from collections import OrderedDict

import PyQt5.QtGui as qtgui
import PyQt5.QtWidgets as qtw
import PyQt5.QtCore as qtcore

import vemos.GUIBackend as guibackend
from vemos.DataRecord import DataRecord

from skimage.measure import compare_ssim, compare_mse, compare_nrmse
from sklearn import svm 
        
class DataSet:
    """
    Represents a set of data records, associated groupings and dissimilarity matrices.

    Parameters
    ----------
    None
    
    Attributes
    ----------
    name : str
        The name of the data set.
    directory_path : str
        The path to the main directory of files.
    description_path : str
        The path to the description file (if needed).
    matrix_directory : str
        The path to where generated matrices are stored.
    matrices : dict of dict
        Dict of each score matrix's name, contents, path, and type.
        Format: {name:{"matrix": matrix_of_scores,
                       "path": matrix_path, 
                       "type": "Similarity"/"Distance/dissimilarity"}...}
    records : list of DataRecord
        The list of records in the data set.
    groupings : dict of {str : dict of {str : list of str}}
        Dict of the groups in each grouping of the data set.
        Format: {grouping: {groupname: [ID1, ID2,...], 
                            groupname: [ID1, ID2,...]...}}
    data_types : OrderedDict of dict of str
        Dict of each data type's name (e.g., "Original Images") type name 
        (e.g., "Image"), file name formats, and file extensions.
        Format: {name: {"type": type_name, 
                        "format": type_format, 
                        "extension": type_ext}...}
    match_nonmatch_scores : dict of {str : dict of {str : list}}
        Dict of the match scores, nonmatch scores, all scores, and ground truth
        values for each (flattened) matrix.
        "gt" stores 1 at index i if the record pair at index i is matched, 
        and 0 otherwise.
        Format: {matrix_name: {"match": [ID1, ...]
                               "nonmatch": [ID2, ...]
                               "all": [ID1, ID2,...]
                               "gt": [1, 0, ...]}}
    
    loading_widget : QWidget
        The widget that displays loadign options.
    icon : QIcon
        The VEMOS icon.
    num_widgets_open : int
        The number of VEMOS widgets open.
    update: bool
        Whether the data set is currently updating.
    interface_to_open : str
        The interface (Data Record Browser or Visual Metric Analyzer) to open.
    has_files : {False, True}
        Whether the data records have associated files.
    gb : GUIBackend
        Instance of GUIBackend for display methods.
    """
    
    def __init__(self):
        self.gb = guibackend.GUIBackend()
        
        self.name = ""
        self.directory_path = ""
        self.description_path = ""
        self.matrix_directory = ""
        self.matrices = {}
        self.records = []
        self.groupings = {"Manual": {}}
        self.data_types = OrderedDict()
        self.data_types["Image"] = {"type": "Image", "format": "*", 
                                    "extension": ".jpg, .png, .tif"} 
        self.data_types["Segmentation"] = {"type": "Segmentation", "format": 
                                           "*_segmentation, *_mask, *_seg", 
                                           "extension": ".jpg, .png, .tif"}
        self.data_types["Curve"] =  {"type": "Curve", "format": "*", 
                                     "extension": ".txt"}                    # Default data types; put all options up here and use to set defaults?
        self.match_nonmatch_scores = {}
        
        self.interface_to_open = "Visual Metric Analyzer"
        self.has_files = True
        self.update = False
        
        self.loading_widget = None
        self.icon = qtgui.QIcon(os.path.join(os.path.dirname(os.path.realpath(__file__)), "glass.ico"))
        self.fig_index = 0
        self.num_widgets_open = 0
        
    
    def create_data_loading_widget(self):
        """ Creates the widget for loading data set information.
        
        The loading widget lets the user select a directory, description file, 
        file types, and/or a score matrix. The directory is stored in 
        `directory_path`, similarity/dissimilarity score matrices are loaded 
        in `matrix_boxes`, and data types are loaded in `data_boxes`.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        int:
            Return code; 0 indicates success and non-zero values indicate an 
            error.
        
        See Also
        --------
        get_records_done(): Executes upon exiting the widget
        
        """
        
        frame_width = 600  
        
        if self.loading_widget:
            self.loading_widget.accept() 
            self.loading_widget.close() 

        self.loading_widget = LoadingWidget()
        self.loading_widget.resize(frame_width, 700)
        self.loading_widget.setWindowTitle("VEMOS")
        self.loading_widget.setWindowIcon(self.icon)
        self.loading_widget.rejected.connect(self.exit_application)
                                                                                
        # Adds title and labels
        font = qtgui.QFont()
        font.setPointSize(12)
        font.setBold(True)    
        
        loading_layout = qtw.QGridLayout()
        
        num_cols = 8

        loading_layout.addWidget(
            self.gb.make_widget(qtw.QLabel("Load Data"), font=font), 0, 0)
        
        loading_layout.addWidget(
            self.gb.make_widget(qtw.QPushButton(self.loading_widget), 
            "clicked", self.load_from_pickle, " Load Previous Session "),
            0, num_cols-1)
            
        loading_layout.setRowMinimumHeight(1, 0)
                 
        font.setPointSize(10)
        loading_layout.addWidget(self.gb.make_widget(qtw.QLabel(
                                 " Set Data Types"), font=font), 7, 0, 1, 5)
        loading_layout.addWidget(self.gb.make_widget(qtw.QLabel(
            " Load Similarity/Dissimilarity Matrices"), font=font), 10, 0, 1, 5)
        
        self.name_line = qtw.QLineEdit(self.name, self.loading_widget)
        
        # Lets user choose directory of files
        self.directory_choice_button = self.gb.make_widget(
            qtw.QPushButton(self.loading_widget), "clicked", 
            self.choose_directory, "Choose File Directory")
        
        self.directory_label = self.gb.make_widget(qtw.QLineEdit(
            self.loading_widget), text=self.directory_path)
        self.directory_label.setReadOnly(True)      
        
                
        # Lets user select how data description information will be loaded
        description_box = qtw.QGroupBox("Load Data Record Files")
            
        self.description_button_group = qtw.QButtonGroup()
        description_layout = qtw.QVBoxLayout()
        
        
        self.directory_description_button = qtw.QRadioButton(
            "Create from Directory", checked=True)
        self.file_description_button = self.gb.make_widget(qtw.QRadioButton(
            "Use Existing Data Description File"))

        
        self.no_files_button = qtw.QRadioButton("No Data Record Files Available")
        
        self.directory_description_button.clicked.connect(partial(
            self.loading_type_change, "directory"))
        self.file_description_button.clicked.connect(partial(
            self.loading_type_change, "description file"))
        self.no_files_button.clicked.connect(partial(
            self.loading_type_change, "no files"))
        
        # Adds buttons to widget
        buttons = [self.directory_description_button, 
                   self.file_description_button, 
                   self.no_files_button]
        for button in buttons:
            description_layout.addWidget(button)
            self.description_button_group.addButton(button)
        
        description_box.setLayout(description_layout)
        
        description_choice_layout = qtw.QHBoxLayout()
        self.description_choice_button = qtw.QPushButton(
            "Choose Description File")
        self.description_choice_button.clicked.connect(
            self.choose_description_file)
        
        self.description_label = self.gb.make_widget(
            qtw.QLineEdit(self.loading_widget), text=self.description_path)
        self.description_label.setReadOnly(True)
        description_choice_layout.addWidget(self.description_choice_button, 1)
        description_choice_layout.addWidget(self.description_label, 3)
                                                      
        self.no_scores_button = qtw.QCheckBox("No Scores Available")                                       
        self.no_scores_button.clicked.connect(partial(
            self.loading_type_change, "no scores"))
        
        # Creates scroll area for loading matrices
        self.matrix_scroll = qtw.QScrollArea(self.loading_widget)
        self.matrix_scroll.setWidgetResizable(True)
        
        self.matrix_scroll_container = qtw.QWidget()
        self.matrix_scroll.setWidget(self.matrix_scroll_container)
        self.matrix_layout = qtw.QVBoxLayout(self.matrix_scroll_container)
        self.matrix_layout.setAlignment(qtcore.Qt.AlignTop)
        
        title_matrix_row = qtw.QHBoxLayout()
        title_matrix_row.addSpacing(85)
        title_matrix_row.addWidget(qtw.QLabel("Name:"))
        title_matrix_row.addSpacing(10)
        title_matrix_row.addWidget(qtw.QLabel("Path:"))
        #title_matrix_row.addSpacing(10)
        title_matrix_row.addWidget(qtw.QLabel(" Matrix Type:"))    
        title_matrix_row.addSpacing(30)
        title_matrix_row.addWidget(qtw.QLabel("  Format:"))                    
        title_matrix_row.addSpacing(50)
        self.matrix_layout.addLayout(title_matrix_row)
        
        #Adds previously loaded matrices, if any
        self.matrix_boxes = []
        path_list = []
        box_row = 0
        csv_row = 0
        for matrix_name in self.matrices:
            
            path_text = self.matrices[matrix_name]["path"]
            type_text = self.matrices[matrix_name]["type"]
            format_text = self.matrices[matrix_name]["format"]
            if format_text=="Matrix" or path_text not in path_list:
                self.add_matrix()
                
                if format_text=="List":
                    self.matrix_boxes[box_row][0].setText("CSV file " + str(csv_row))
                    csv_row += 1
                    path_list.append(path_text)
                else:
                    self.matrix_boxes[box_row][0].setText(matrix_name)
                    
                self.matrix_boxes[box_row][1].setText(path_text)
                self.matrix_boxes[box_row][2].setCurrentText(type_text)
                self.matrix_boxes[box_row][3].setCurrentText(format_text)
                box_row += 1
            
        self.add_matrix()
        
        # Lets user select what GUI to open
        gui_choice_layout = qtw.QVBoxLayout()
        
        self.analyzer_button = self.gb.make_widget(
            qtw.QRadioButton("Open Visual Metric Analyzer"), "toggled", 
            self.interface_type_change)
        gui_choice_layout.addWidget(self.analyzer_button)
        
        self.browser_button = self.gb.make_widget(
            qtw.QRadioButton("Open Data Record Browser"), "toggled", 
            self.interface_type_change)
        gui_choice_layout.addWidget(self.browser_button)
        
        if self.interface_to_open == "Data Record Browser":
            self.browser_button.setChecked(True)
        else:
            self.analyzer_button.setChecked(True)
            
        gui_choice_box = qtw.QGroupBox("Interface to Open")
        gui_choice_box.setLayout(gui_choice_layout) 
        
        # Lets user select data types
        
        # Lets user select how score matrices (if any) will be loaded        
        self.use_folders_button = qtw.QCheckBox("Read Data Types from Folders")                                       
        self.use_folders_button.clicked.connect(partial(
            self.loading_type_change, "use folders"))
        
        # Adds scroll area for data types
        self.data_type_scroll = qtw.QScrollArea(self.loading_widget)
        self.data_type_scroll.setWidgetResizable(True)
        
        self.data_type_scroll_container = qtw.QWidget()
        self.data_type_scroll.setWidget(self.data_type_scroll_container)
        self.data_type_layout = qtw.QVBoxLayout(
            self.data_type_scroll_container)
        self.data_type_layout.setAlignment(qtcore.Qt.AlignTop)
                                 
        title_row = qtw.QHBoxLayout()
        title_row.addWidget(qtw.QLabel("Type:"))
        title_row.addWidget(qtw.QLabel("Name:   "))
        title_row.addSpacing(20)  
        title_row.addWidget(qtw.QLabel("Format:"))
        title_row.addSpacing(80)                      
        title_row.addWidget(qtw.QLabel("   Extension:"))
        title_row.addSpacing(65)
        self.data_type_layout.addLayout(title_row)
        
        self.data_boxes = []
        
        # Adds previously selected data types, if any
        for row, data_type_name in enumerate(self.data_types):
            self.add_data_type()
            row_data = self.data_types[data_type_name]
            type_index = self.data_boxes[row][0].findText(row_data["type"])
            self.data_boxes[row][0].setCurrentIndex(type_index)
            self.data_boxes[row][1].setText(data_type_name)
            self.data_boxes[row][2].setEditText(row_data["format"])
            self.data_boxes[row][3].setEditText(row_data["extension"])
        
        # Adds buttons
        button_row = qtw.QHBoxLayout()
        button_row.setSpacing(20)        
        button_row.addWidget(self.gb.make_widget(qtw.QPushButton(
            " Add Another Score Matrix "), "clicked", self.add_matrix))
        button_row.addStretch(2)
        loading_layout.setRowMinimumHeight(1, 20)
        button_row.addWidget(self.gb.make_widget(qtw.QPushButton(
            "   Cancel   "), "clicked", self.cancel))
        button_row.addWidget(self.gb.make_widget(qtw.QPushButton(
            "   Done   "), "clicked", self.get_records_done))

        # Organizes widgets in grid
        loading_layout.setVerticalSpacing(10)
        loading_layout.addWidget(self.gb.make_widget(qtw.QLabel(
            "Data Set Name: ")), 2, 0)                               
        loading_layout.addWidget(self.name_line, 2, 1, 1, num_cols-1)                                             
        loading_layout.addWidget(self.directory_choice_button, 3, 0)                                             
        loading_layout.addWidget(self.directory_label, 3, 1, 1, num_cols-1)   
        loading_layout.addWidget(gui_choice_box, 4, 0, 1, 2)                                          
        loading_layout.addWidget(description_box, 5, 0, 1, 2)
        loading_layout.addLayout(description_choice_layout, 5, 2, 1, 6)
        
        loading_layout.addWidget(self.use_folders_button, 7, num_cols-1)
        loading_layout.addWidget(self.data_type_scroll, 8, 0, 1, num_cols)
        loading_layout.addWidget(self.gb.make_widget(qtw.QPushButton(
            " Add Another Data Type "), "clicked", self.add_data_type), 9, 0)
        loading_layout.addWidget(self.no_scores_button, 10, num_cols-1)
        loading_layout.addWidget(self.matrix_scroll, 11, 0, 1, num_cols)
        loading_layout.addLayout(button_row, 12, 0, 1, num_cols)
        
        self.loading_widget.setLayout(loading_layout)
        
        if self.has_files == False:
            self.analyzer_button.setChecked(True)
            self.no_files_button.setChecked(True)
            self.interface_type_change(self.browser_button)
            self.loading_type_change("no files", self.no_files_button)    
            
        if self.description_path != "":
            self.directory_description_button.setChecked(False)
            self.file_description_button.setChecked(True)
            self.loading_type_change("description file")
        else:
            self.loading_type_change("directory")
            
        return self.loading_widget.exec_()
    
    
    def interface_type_change(self, button):
        """ Disables/enables options for the Record Browser and Data Analyzer.
        
        The Analyzer requires a score matrix and the Browser requires data 
        files, so this method disables the options to load without a score 
        matrix when "Open Visual Metric Analyzer" is clicked and disables the 
        option to load without data files when "Open Record Browser" is 
        clicked.
        
        Parameters
        ----------
        button : QButton
            The button clicked.
        
        Returns
        -------
        None
        
        """
        
        # For the Visual Metric Analyzer, requires at least one score matrix
        if button.text() == "Open Visual Metric Analyzer":
            self.no_files_button.setEnabled(True)
            self.no_scores_button.setEnabled(False)
            self.no_scores_button.setChecked(False)
            self.matrix_scroll_container.setEnabled(True)
        
        # For the Record Browser, requires data files        
        else:
            self.no_files_button.setEnabled(False)
            self.no_scores_button.setEnabled(True)
            
            if self.no_files_button.isChecked():

                self.description_button_group.setExclusive(False)
                self.no_files_button.setChecked(False)
                self.description_button_group.setExclusive(True)
                self.data_type_scroll_container.setEnabled(True)

                self.description_path = ""
    
    
    def loading_type_change(self, loading_method, button=None):
        """ Disables/enables options when the user chooses how to load 
        scores/files.
        
        If the user chooses not to load score matrices, disables the score 
        matrix loading area; if he/she chooses to load via folders, disables 
        the option to select a description file; and if he/she chooses to load 
        data files via a directory, use a description file, or not load them 
        at all, disables the other two options.
        
        Parameters
        ----------
        loading_method : str
            The option selected by the user ("no scores", "use folders", "no 
            files", "directory", or "description file").
        
        button : QButton, optional
            The button clicked.
        
        Returns
        -------
        None
        
        """
        
        # Disables or enables score matrix loading
        if loading_method == "no scores":
            if self.no_scores_button.isChecked():
                self.matrix_scroll_container.setEnabled(False)
            else:
                self.matrix_scroll_container.setEnabled(True)
        
        # Disables or enables loading via folders (versus file formatting)
        elif loading_method == "use folders":
            for row in self.data_boxes:

                if self.use_folders_button.isChecked():
                    row[2].setEnabled(False)
                    row[3].setEnabled(False)
                else:
                    row[2].setEnabled(True)
                    row[3].setEnabled(True)
                        
        # Disables or enables data type and description file selection
        else:
            if loading_method == "description file":
                self.data_type_scroll_container.setEnabled(True)
                self.description_choice_button.setEnabled(True)
                self.description_label.setEnabled(True)
                description_text = self.description_label.text()
                self.description_path = str(description_text)
                self.has_files=True
                
                # Disables data type formatting if using description file
                for row in self.data_boxes:
                    row[2].setEnabled(False)
                    row[3].setEnabled(False)
                    
            else:
                for row in self.data_boxes:
                    row[2].setEnabled(True)
                    row[3].setEnabled(True)
                
                if loading_method == "no files":
                    self.data_type_scroll_container.setEnabled(False)
                    self.description_choice_button.setEnabled(False)
                    self.description_label.setEnabled(False)
                    self.has_files=False
            
                elif loading_method == "directory":
                    self.data_type_scroll_container.setEnabled(True)
                    self.description_choice_button.setEnabled(False)
                    self.description_label.setEnabled(False)
                    self.has_files=True
                
            
    def add_matrix(self):
        """Adds a row of QComboBoxes to "Load Similarity/Dissimilarity Scores."
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        """
        
        x1 = 50
        xint = 120
        y1 = 50
        yint = 30
        
        num_matrices = len(self.matrix_boxes)
        row = qtw.QHBoxLayout()
        
        choose_button=qtw.QPushButton(self.loading_widget)
        choose_button.setText("Choose")
        choose_button.clicked.connect(partial(
            self.choose_score_matrix, choose_button))
        row.addWidget(choose_button)

        name_box = self.gb.make_widget(
            qtw.QLineEdit(), pos=(x1, y1 + (1+num_matrices)*yint + 4), 
            resizing=(30, 25))
        row.addWidget(name_box)
                                            
        path_box = self.gb.make_widget(
            qtw.QLineEdit(), pos=(x1 + xint, y1 + (1+num_matrices)*yint + 4), 
            resizing=(30, 25))     
        path_box.setReadOnly(True)
        row.addWidget(path_box) 
                                       
        type_box = self.gb.make_widget(
            qtw.QComboBox(), 
            pos=(x1 + 2.5*xint, y1 + (1+num_matrices)*yint + 4), 
            items=["Distance/Dissimilarity", "Similarity"]) 
        row.addWidget(type_box)
        
        format_box = self.gb.make_widget(
            qtw.QComboBox(), 
            pos=(x1 + 3*xint, y1 + (1+num_matrices)*yint + 4), 
            items=["Matrix", "List"]) 
        row.addWidget(format_box)
        
        row.addWidget(self.gb.make_widget(
            qtw.QPushButton(self.loading_widget), "clicked", 
            partial(self.remove_row, row, self.matrix_boxes), "Remove"))
        
        self.matrix_layout.addLayout(row)
        self.matrix_boxes.append([
            name_box, path_box, type_box, format_box, choose_button])
    
        
    def add_data_type(self):
        """ Adds another row of combo boxes to "Set Data Types." 
        
        The options for each row default to the data types, file name formats, 
        and file extensions in `types`, `formats`, and `extensions`, 
        respectively.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        """
        
        x1 = 50
        xint = 145
        y1 = 50
        yint = 30
        
        # Default names, formats, and extensions
        types = ["Image", "Segmentation", "Curve", "Description", 
                 "Point Cloud"]
        formats = ["*", "*_segmentation, *_mask, *_seg", "*", "*_description", 
                   "*_cloud, *_points", "*_features"]
        extensions = [".jpg, .png", ".jpg, .png", ".txt", ".txt", ".txt"]
        
        num_extensions = len(self.data_boxes)
        row = qtw.QHBoxLayout()
        
        type_box = self.gb.make_widget(
            qtw.QComboBox(), "currentIndexChanged", 
            partial(self.data_type_choice, num_extensions), 
            pos=(x1, y1 + (1+num_extensions)*yint + 4), items=types)                # Update make_widg to do this in there
        row.addWidget(type_box)
        
        name_box = self.gb.make_widget(
            qtw.QLineEdit(), pos=(x1, y1 + (1+num_extensions)*yint + 4))
        row.addWidget(name_box)
                                            
        format_box = self.gb.make_widget(
            qtw.QComboBox(), pos=(x1 + xint, 
            y1 + (1+num_extensions)*yint + 4), items=formats, editable=True)     
        row.addWidget(format_box) 
                                       
        extension_box = self.gb.make_widget(
            qtw.QComboBox(), pos=(x1 + 2.5*xint, 
            y1 + (1+num_extensions)*yint + 4), items=extensions, editable=True) 
        row.addWidget(extension_box)
        
        # Disables formatting options if not needed
        if (self.use_folders_button.isChecked() or 
             self.file_description_button.isChecked()):
            extension_box.setEnabled(False)
            format_box.setEnabled(False)
        
        row.addWidget(self.gb.make_widget(
            qtw.QPushButton(self.loading_widget), "clicked", 
            partial(self.remove_row, row, self.data_boxes), "Remove"))
        
        self.data_type_layout.addLayout(row)
        self.data_boxes.append([type_box, name_box, format_box, extension_box])
            
         
    def data_type_choice(self, extension_index, data_type_index):
        """ Changes the default formats/extensions when a data type is selected.
        
        Parameters
        ----------
        extension_index : int
            The index of the changed row in `data_boxes`.
        
        data_type_index : int
            The index of the selected data type in the list of data types.
        
        Returns
        -------
        None
        
        See Also
        --------
        add_data_type : More details on the format of `data_boxes`
        
        """
        
        cur_type_text = self.data_boxes[extension_index][0].currentText()
        self.data_boxes[extension_index][1].setText(cur_type_text)
        self.data_boxes[extension_index][2].setCurrentIndex(data_type_index)
        self.data_boxes[extension_index][3].setCurrentIndex(data_type_index) 
    
    
    def remove_row(self, row, remove_from):
        """ Removes a row from the list of matrices or data types.
        
        Parameters
        ----------
        row : int
            The index of the changed row in `data_boxes`.
        
        remove_from : list of list
            The list of rows (self.data_boxes or self.matrix_boxes) from which 
            to remove a row.
        
        Returns
        -------
        None
        
        """
    
        # Finds row index   
        name_index = 1
        if remove_from == self.data_boxes:
            name_index = 0

        index = 0
        for index in range(len(remove_from)):
            if row.itemAt(name_index).widget() == remove_from[index][0]:
                del remove_from[index]
                break
            
        # Removes row from widget
        for i in range(5 + name_index):                                 
            row.itemAt(i).widget().close()

        if remove_from == self.data_boxes:
            self.data_type_layout.removeItem(row)
        else:
            self.matrix_layout.removeItem(row)
        
    
    def get_records_done(self):
        """ Upon closing the data loading widget, reads in and stores data.
        
        Adds the data types in `data_boxes` to `data_types`, adds the 
        score matrices in `matrix_boxes` to `matrices`, and loads 
        records from folder or description file information into 
        `records`. Reopens the loading widget and does not store the 
        data if information is missing, if data type names overlap, or 
        if any loading method fails.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        See Also
        --------
        create_data_loading_widget : Creates the widget for data loading that 
            was closed
        load_matrices : Loads the score matrices
        load_records_from_directory : Loads data records from a file directory
        load_records_from_description_file : Loads data records from a 
            description file containing the contents of each record
        
        """
        
        # If information is missing, warns user and leaves loading widget open
        if not self.directory_path:
            self.gb.loading_error("directory")
            return
        
        name_text = self.name_line.text()
        if name_text == "":
            self.gb.loading_error("name for the data set")
            return
        else:
            self.name = name_text
        
        # Reads in score matrices
        if self.load_matrices() != True:
            self.clear_data()
            return
        
        # Retrieves data types
        self.data_types = OrderedDict()
        if not self.no_files_button.isChecked():
            for row in self.data_boxes:
    
                data_type = []
                for widget in row:
                    if type(widget) is qtw.QComboBox:
                        data_type.append(str(widget.currentText()))
                    else:
                        data_type.append(str(widget.text()))
    
                # If data type name is not repeated, adds to list of data types
                if data_type[1] not in self.data_types:
                    self.data_types[data_type[1]] = {"type": data_type[0], 
                                                     "format": data_type[2], 
                                                     "extension": data_type[3]}
                elif self.use_folders_button.isChecked():
                    self.gb.general_msgbox(
                        "Data Type Error", ("To load from a directory based on" 
                        " folder naming, please ensure that all data types are"
                        " in folders with unique names."))
                    return

        # Retrieves data records from description file or directory
        if self.directory_description_button.isChecked():
            if self.load_records_from_directory(self.use_folders_button.isChecked()) != True:
                self.clear_data() 
                return
                
            self.description_path = ""
                
        elif not self.no_files_button.isChecked():
            if not self.description_path:
                self.gb.loading_error("data description file")
                self.clear_data() 
                return
                
            if self.load_records_from_description_file() != True:
                self.clear_data()               
                return

        # Loads any scores from CSV files
        #if self.use_csv_button.isChecked():
        if self.load_score_lists() != True:
            self.clear_data()
            return
        if (not self.no_scores_button.isChecked()) and len(self.matrices)==0:
            self.gb.loading_error("similarity/dissimilarity matrix")
            self.clear_data()
            return
            
        #Determines which interface to open
        if self.analyzer_button.isChecked():
            self.interface_to_open = "Visual Metric Analyzer"
        else:
            self.interface_to_open = "Data Record Browser"
        
        # Checks that the length of the matrices equals the number of records
        for matrix in self.matrices:
            if len(self.matrices[matrix]["matrix"]) != len(self.records):
                self.gb.general_msgbox(
                    "Loading Error", "The size of the matrix " + matrix + 
                    " does not match the number of records.")
                self.clear_data()
                return
        
        if self.get_match_scores()==False:
            self.clear_data()
            return
        
        
        # If no files and not CSV loading, makes placeholder records
        if len(self.records)==0:
            if len(self.matrices)==0 or not(self.no_files_button.isChecked()):
                self.gb.loading_error("data set")
                self.clear_data()
                return
            
            test_matrix = next(iter(self.matrices.values()))
            for index in range(len(test_matrix["matrix"][0])):
                self.add_new_record(str(index), [])
            
        self.loading_widget.accept()
        #self.print_all_data()
        self.loading_widget.close()

    
    def clear_data(self):
        """ In case of a data loading error, removes partially loaded data.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None

        See Also
        --------
        get_records_done : Calls clear_data if any loading methods fail
        
        """
        self.matrices = {}
        self.records = []
        self.groupings = {"Manual": {}} 
        self.match_nonmatch_scores = {}
        self.data_types = OrderedDict()
    
    
    def exit_application(self):
        """ Closes the program when the user selects "Cancel".
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        sys.exit(-1)
        
  
    
#################################################################################
# Methods for loading and storing directory, records, and scores    
    
    
    def choose_directory(self, button=None):
        """ Loads and stores the directory path.
        
        Parameters
        ----------
        button : QButton, optional
            The button clicked (`directory_choice_button`).
        
        Returns
        -------
        None
        
        See Also
        --------
        fix_path : Corrects irregular use of slashes under Windows OS.
        """
        
        try:
            original_directory = qtw.QFileDialog.getExistingDirectory(
                None, "Select Data File Directory")
            main_directory = self.fix_path(str(original_directory))                  
        except IOError:
            self.gb.loading_error("directory")
            return
        
        self.directory_path = main_directory
        self.directory_label.setText(self.directory_path)
        if self.matrix_directory == "":
            self.matrix_directory = os.path.join(self.directory_path, 
                                                 "_generated_matrices")
        
        if self.name_line.text() == "":
            name_index = self.directory_path.rfind("\\")+1
            self.name_line.setText(self.directory_path[name_index:])
        
     
    def choose_description_file(self, button=None):    
        """ Loads and stores the path to the description file.
        
        Parameters
        ----------
        button : QButton, optional
            The button clicked (`description_choice_button`).
        
        Returns
        -------
        None
        
        """
        
        try:
            if self.directory_path:
                description = self.fix_path(str(
                    qtw.QFileDialog.getOpenFileName(
                    caption="Open Data Description File", 
                    directory=self.directory_path)[0]))
            else:
                description = self.fix_path(str(
                    qtw.QFileDialog.getOpenFileName(
                    caption="Open Data Description File")[0]))                        
        except (IOError, ValueError):
            self.gb.loading_error("data description file")
            return
            
        self.description_path = description
        self.description_label.setText(self.description_path)
    
    
    def choose_score_matrix(self, button=None):
        """ Updates row with the name and path of newly chosen score matrix.
        
        Parameters
        ----------
        button : QButton, optional
            The button clicked ("Choose").
        
        Returns
        -------
        None

        """
        try:
                       
            for row_index, test_row in enumerate(self.matrix_boxes):
                if test_row[4]==button:
                    break
            else:
                raise ValueError
        
            title_text = "Open Data Matrix"
            if self.directory_path:
                fname=qtw.QFileDialog.getOpenFileName(
                    caption=title_text, directory=self.directory_path)                
            else: 
                fname=qtw.QFileDialog.getOpenFileName(
                    caption=title_text)
            
            if fname is None or fname[0]=='':
                return
            
            matrix = self.fix_path(str(fname[0]))
            
        except (IOError, ValueError):                    
            self.gb.loading_error("similarity/dissimilarity matrix")
            return
        
        matrix_name = matrix.split("\\")[-1].split(".")[0]        

        self.matrix_boxes[row_index][0].setText(matrix_name)
        self.matrix_boxes[row_index][1].setText(matrix)
    
    
    def load_records_from_description_file(self):
        """ Creates a set of data records from the given text file.
        
        The description file should be a text file with each row in the format:
        id; (group1, group2...); (match1, match2...); filepath1; filepath2...
        Matches should be mutual (e.g., if ID A matches ID B, ID B should match
        ID A). Otherwise, the missing matches are added and the user may save 
        the new description file.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        bool :
            True indicates success; False indicates error.
        
        """

        self.records = []
        self.groupings = {"Original Level 1": {}, "Manual": {}} 
        
        try:
            description_file = open(str(self.description_path))
            reader = csv.reader(description_file, delimiter=';')
            
            for line in reader:
                row = [item.strip() for item in line]
                if len(row) != len(self.data_types) + 3:
                    raise IOError
        
                # Gets lists of groups and matches and dictionary of file names
                group_list = re.sub('[\(\)]', "", row[1]).split(",")
                cur_groups = [item.strip() for item in group_list if item!=""]
                match_list = re.sub('[\(\)]', "", row[2]).split(",")
                cur_matches = [item.strip() for item in match_list if item!=""]
                cur_files = {}
                
                for item in row[3:]:
                    if item and not item.isspace():                     # Should work for either leaving in "Images: " when there's no image or just leaving a space, but needs to be in paper
                        data_type, cur_path = item.split(":")
                        cur_path = cur_path.strip()
                        data_type = data_type.strip()
                        
                        if data_type not in self.data_types:
                            raise IOError
                        if cur_path and not cur_path.isspace():     
                            cur_files[data_type] = self.fix_path(cur_path)

                
                # Creates record
                new_record = DataRecord(row[0], cur_groups, cur_matches, 
                                        cur_files)
                self.records.append(new_record)
                
                # Updates dictionary of groups                                  Update for subgroups
                for group in cur_groups:
                    if group not in self.groupings["Original Level 1"]:
                        self.groupings["Original Level 1"][group]=[row[0]]
                    else:
                        self.groupings["Original Level 1"][group].append(row[0])

            # Ensures that matches are mutual
            result = None
            for record in self.records:
                for match in record.matches:
                    matches_of_match = self.record_from_id(match).matches

                    if record.id not in matches_of_match:
                        self.record_from_id(match).matches.append((record.id))

                        if result is None:
                            result = self.gb.general_msgbox(
                                "Non-Mutual Match Found", ("This description "
                                "file contains non-mutual matches. Would you "
                                "like to save a corrected description file?"), 
                                cancel=True)                                    # If they don't, make sure to add an *
            
            if result==0:
                self.save_description_file_as()
                
            return True
            
        except (IOError, ValueError, csv.Error):
            self.gb.loading_error("data description file")
            self.records = []
            self.groupings = {"Manual": {}} 
            return False                                                        # change all of these to T/F
        
    
    def load_records_from_directory(self, dtypes_in_folders=False, ids_in_folders=False):                 # Add button for this option (NOT same as data types in folders; this is only if ID names are the folders)
        """ Makes a set of data records from the files in the stored directory.
        
        Checks that the format + extension combination for each data type is 
        unique. Identifies group, data type, and ID information from the file 
        path. If two IDs are the same, but correspond to different records 
        (e.g., because they are in different groups), changes all IDs to the 
        format group_ID.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        int :
            Non-zero value indicates error code, zero indicates success.
        
        See Also
        --------
        add_new_record : Adds a new record to `records`
        
        Notes
        -----
        Data should be cleared if this method returns False.
        
        """
        
        self.records = []
        self.groupings = {"Manual": {}}
        cur_data_type = ""
        cur_groups = []
        cur_id = ""
        id_group_level = None # Check whether there are different groups with same IDs, which should be saved as different records
        
        # Checks that all combinations of format + extension are different
        if not dtypes_in_folders:
            for row, data_type in enumerate(self.data_types):                            # Check efficiency
            
                all_exts = set(self.data_types[data_type]["extension"].split(", "))
                all_formats = set(self.data_types[data_type]["format"].split(", "))
                
                for cur_row in range(row):                                            # But what if they're using named subdirectories? Update documentation in that case
                    
                    cur_keys = list(self.data_types.values())[cur_row]
                    cur_extension = set(cur_keys["extension"].split(", "))
                    extension_overlap = len(all_exts.intersection(cur_extension))
                    cur_format = set(cur_keys["format"].split(", "))
                    format_overlap = len(all_formats.intersection(cur_format))
                    
                    if extension_overlap != 0 and format_overlap != 0:
                        self.gb.general_msgbox(
                            "Data Type Error", ("To load from a directory, please "
                            "ensure that all data types have a unique combination "
                            "of file name format and extension."))
                        return False
        
        # Loads in all files in the directory
        try:
            all_formats = []
            all_extensions = []
            
            # Re-formats data types to work with regex            
            format_changes = {" ": "", "*": ".*", ",": "|"}
            for data_type in self.data_types:
                formats = self.data_types[data_type]["format"]
                for original in format_changes:
                    formats = formats.replace(original, format_changes[original])
                formats = formats.split("|")
                
                ext_text = self.data_types[data_type]["extension"]
                extensions = ext_text.split(", ")  
                
                all_formats.append(formats)
                all_extensions.append(extensions)
     
            
            for root, directories, filenames in os.walk(self.directory_path):
                
                # Look at all folders not within where matrices are stored, unless matrices are stored outside the main folder
                if not (((self.matrix_directory in root) and (self.directory_path in self.matrix_directory)) or root.endswith("_generated_matrices")):
                    
                    # Checks for subdirectories and files on the same levels
                    if len(directories) != 0 and len(filenames) != 0:
                        self.gb.general_msgbox(
                            "Error", "Error: Files on same level as directories")
                        raise IOError
                    
                    for cur_file in filenames:
                        root = self.fix_path(root)
                        full_path = os.path.join(root, cur_file)
                        
                        # Gets the full path to file from the main directory
                        start_path, end_path = full_path.split(
                            self.directory_path)
                        end_path = end_path.strip("\\")
                        
                        # Determines the current group, id, or data type
                        cur_data_type = ""
                        cur_groups = []
                        cur_id = ""
                        folders = re.split(r'\\|/', end_path)[:-1]
                        
                        for level, folder in enumerate(folders):    
                            
                            # Checks if data type/group/ID information in file path
                            if folder in self.data_types:                             # Might be substring of root
                                cur_data_type = folder
                            
                            elif folder[-1]=="s" and folder[:-1] in self.data_types:         #OJO
                                cur_data_type = folder[:-1]
                            
                            # To be an ID, must either a) be at bottom level or b) have only data type levels after it
                            elif ids_in_folders and (level == len(folders)-1 or
                                 folders[level + 1] in self.data_types):                                               # Remember to clear
                                cur_id = folder
                            
                            # Otherwise, it's a group
                            else:
                                cur_groups.append(folder)                
                        
                        # Identifies the data type and ID of a file, if necessary
                        if cur_data_type == "" or cur_id == "":
        
                            (best_data_type_match, best_format_match) = self.get_best_data_type_match(cur_file, all_formats, all_extensions)
                            
                            if cur_data_type == "":
                                cur_data_type = best_data_type_match
                            if cur_id == "":
                                cur_id = re.sub(best_format_match.replace(
                                    ".*", ""), "", 
                                    cur_file[:cur_file.rfind(".")])
                        
    
                        # Checks whether all necessary information has been loaded
                        if cur_id == "" or cur_data_type == "" or cur_groups == []:
                            raise IOError
                            
                        # Adds file to record; creates new record if necessary
                        if id_group_level is not None:
                            cur_id = cur_groups[id_group_level] + "_" + cur_id
                        
                        matching_record = self.record_from_id(cur_id)
                        if matching_record != -1:
                            
                            # Checks if a different record (same id, but different group)
                            if ( id_group_level is not None or 
                                 len(matching_record.groups) != len(cur_groups)):
                                
                                matching_record.files[cur_data_type] = end_path
                            
                            else:
                                
                                # If different groups are using the same IDs, changes all IDs
                                for index, match_group in enumerate(
                                    matching_record.groups):  

                                    if match_group != cur_groups[index]:
                                        
                                        for record in self.records:
                                            record.id = record.groups[index] + "_" + record.id
                                            
                                        id_group_level = index
                                        cur_id = cur_groups[index] + "_" + cur_id
                                        self.add_new_record(cur_id, cur_groups, end_path, cur_data_type)
    
                                        break
                                else:
                                    matching_record.files[cur_data_type] = self.fix_path(end_path)
         
                        # Otherwise, adds a new record
                        else:
                            self.add_new_record(cur_id, cur_groups, end_path, 
                                                cur_data_type)           
                
        except (IOError, ValueError):
            self.gb.loading_error("directory")
            self.groupings = {"Manual": {}} 
            return False
        
        return True

    
    def add_new_record(self, new_id, new_groups, end_path=None, 
                       new_data_type=None):
        """ Adds a new record to `records`.
        
        Parameters
        ----------
        new_id : str
            The ID of the record to add.
        new_groups : list of str
            The list of groups of the record.
        end_path : str, optional
            The path to the first data file in the record, excluding the 
            path up to the main file directory.
        new_data_type : str, optional
            The data type of the file found at end_path.
        
        Returns
        -------
        None
        
        See Also
        --------
        load_records_from_directory: Loads a set of records from a file folder. 
        """
        
        new_record_files = {}
        if new_data_type:
            new_record_files[new_data_type] = self.fix_path(end_path)
        self.records.append(DataRecord(new_id, new_groups, [], 
                                       new_record_files))
        
        # Adds ID to its group; adds new group if needed
        for index, group in enumerate(new_groups):
            grouping_name = "Original Level " + str(index+1)
            
            if grouping_name not in self.groupings:
                self.groupings[grouping_name] = {}
                
            if group in self.groupings[grouping_name]:
                self.groupings[grouping_name][group].append(new_id)
            else:
                self.groupings[grouping_name][group] = [new_id]
        
    
    # 
    def get_best_data_type_match(self, cur_file, all_formats, all_extensions):
        """ Returns the best format and extension match for `cur_file`.
        
        Parameters
        ----------
        cur_file : str
            The name of the file (including the extension).
        all_formats : list of list of str
            The list of all acceptable file name formats for each data type.
        all_extensions : list of list of str
            The list of all acceptable file extensions for each data type.
        
        Returns
        -------
        best_data_type_match : str
            The data type that best matches the file name format.
        best_format_match : str
            The format that resulted in the best match.
        
        See Also
        --------
        load_records_from_directory: Loads a set of records from a file folder, 
            using the best data type match to determine their data types. 
        """        
        
        best_data_type_match = ""
        best_format_match = ""
        
        file_minus_ext = cur_file[:cur_file.rfind(".")]
        for index, data_type in enumerate(self.data_types):
            for extension in all_extensions[index]:
                if cur_file.lower().endswith(extension.lower()):
                    
                    # Checks that format doesn't match another data type
                    for cur_format in all_formats[index]:
                        if re.match(cur_format, file_minus_ext): 
                            
                            if (best_data_type_match == "" or 
                                len(cur_format) > len(best_format_match)):
                                best_data_type_match = data_type
                                best_format_match = cur_format
                    
        return (best_data_type_match, best_format_match)
    
            
    def load_matrices(self):
        """ Loads the score matrices in `matrix_boxes` into `matrices`.
        
        Checks that matrix names are unique and symmetrizes the matrices if 
        necessary. This method loads only the matrices in matrix format.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        bool:
            True if matrices were loaded successfully, and False otherwise.

        See Also
        --------
        load_score_lists : Loads matrices in list format.
        """ 
              
        # Reads in score matrices
        self.matrices = {}
        if not self.no_scores_button.isChecked():

            # For each matrix, gets name and file path, then reads in scores
            for row in self.matrix_boxes:
                if str(row[3].currentText()) == "Matrix":
                    
                    # Checks if loaded via combo boxes or text file
                    if isinstance(row[0], str):
                        new_matrix_name = row[0]
                        new_matrix = {"path": row[1], "type": row[2]}
                    else:
                        new_matrix_name = str(row[0].text())                
                        new_matrix = {"path": str(row[1].text()), 
                                      "type": str(row[2].currentText()),
                                      "format": "Matrix"}
                                      
                    if new_matrix_name != "":
                        
                        # If matrix name not repeated, adds to list of matrices
                        if new_matrix_name not in self.matrices:
                            self.matrices[new_matrix_name] = new_matrix
                        else:
                            self.gb.general_msgbox(
                                "Matrix Naming Error", ("Please ensure that "
                                "all score matrices have unique names."))
                            self.matrices = {}
                            return False
                        
                        try:
                            new_matrix["matrix"] = np.loadtxt(
                                new_matrix["path"])
                            
                            # Symmetrizes the matrix if necessary
                            if not np.allclose(new_matrix["matrix"], 
                                               new_matrix["matrix"].T):
                                
                                items = ("Use max([i, j], [j,i])", 
                                         "Use min([i, j], [j,i])", 
                                         "Use average", 
                                         "Use (i, j) value", 
                                         "Use (j, i) value", 
                                         "Skip this matrix")
    
                                sym_text = ("Symmetric matrices are required, "
                                    "but the matrix " + new_matrix_name + " is"
                                    " not symmetric. Would you like to "
                                    "symmetrize it?")
                                item, ok = qtw.QInputDialog.getItem(
                                    self.loading_widget, "Asymmetric Matrix", 
                                    sym_text, items, 0, False)
                                
                                item = items.index(item)
                                if ok == False:
                                    return False
                                if item == 5:
                                    del self.matrices[new_matrix_name]
                                else:
                                    new_matrix["matrix"] = self.symmetrize_matrix(new_matrix["matrix"], item)
                            
                            
                        except (IOError, ValueError, SyntaxError):
                            self.gb.loading_error(
                                "similarity/dissimilarity matrix. The matrix " 
                                + new_matrix_name + " failed to load")
                            return False
            
        return True

    def load_score_lists(self):
        """ Loads score lists in `matrix_boxes` into `matrices`.
        
        This method loads only the scores in list format. Also creates the list
        of `records` if not loaded via directory or description file at this 
        point.
        
        Lists should be in the format:
                        Metric_1    Metric_2 ...   Ground Truth
        ID1     ID2     [score]     [score]  ...    Y
        ID6     ID8     [score]     [score]  ...    N
        ...
        
        Parameters
        ----------
        None
        
        Returns
        -------
        bool:
            True if matrices were loaded successfully, and False otherwise.
            
        """
        
        if not self.no_scores_button.isChecked():

            # Examines lists only
            for box_row in self.matrix_boxes:
                if str(box_row[3].currentText()) == "List":
                
                    # Checks if loaded via combo boxes or text file
                    if isinstance(box_row[0], str):
                        file_path=box_row[1]
                    else:
                        file_path=str(box_row[1].text())
                      
                    if file_path != "":
                        try:
                            with open(file_path, 'r') as csv_scores:
                                reader = csv.reader(csv_scores)
                                scores_0 = list(reader)
                                
                                # Makes the records, if no directory or 
                                # description available
                                if len(self.records) == 0:
                                    record_list = []
                                    for row in scores_0[1:]:
                                        if row[0] not in record_list:
                                            record_list.append(row[0])
                                        if row[1] not in record_list:
                                            record_list.append(row[1])
            
                                    for record_name in record_list:
                                            self.records.append(DataRecord(
                                                record_name, [], [], {}))
                                    
                                # Get matrix names
                                for item in scores_0[0][2:-1]:
                                    new_matrix=np.zeros((len(self.records), 
                                                         len(self.records)))
                                    new_matrix.fill(None)
                                    
                                    if item not in self.matrices:
                                        self.matrices[item] = {
                                            "path": file_path, 
                                            "type": str(
                                                box_row[2].currentText()), 
                                            "matrix": new_matrix, 
                                            "format": "List"}
                                        
                                    else:
                                        self.gb.general_msgbox(
                                            "Matrix Naming Error", 
                                            ("Please ensure that all score "
                                             "matrices have unique names."))

                                        return False
                                 
                                # Ensure that matrix order matches record order
                                for row in scores_0[1:]:
                                    if len(row)<2:
                                        raise IndexError
                                        
                                    index1=self.index_from_id(row[0])
                                    index2=self.index_from_id(row[1])
                                    
                                    if (index1 == -1 or index2 == -1 or 
                                        index1 > len(self.records) or 
                                        index2 > len(self.records)):
                                        
                                        raise IndexError
                                    
                                    for col, item in enumerate(row[2:-1]):
                                        if item != '':
                                            mtx_name = scores_0[0][col+2]
                                            self.matrices[mtx_name]["matrix"][
                                                index1][index2] = item
                                    
                                    # Add ground truth values as matches 
                                    # if available and needed
                                    if row[-1]=="Y" or row[-1]=="y":
                                        record1=self.record_from_id(row[0])
                                        record2=self.record_from_id(row[1])
                                        if row[1] not in record1.matches:
                                            record1.matches.append(row[1])
                                            record2.matches.append(row[0])
                                
                        except (IOError, OSError, ValueError, IndexError):
                            self.gb.loading_error("CSV file")
                            return False
                    
        return True


    def load_from_pickle(self):
        """ Loads a previously pickled data set. 
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
            
        See Also
        --------
        save: Pickles the current data set.
        """
        
        # Checks what pickle file to open
        folder = os.path.dirname(os.path.realpath(__file__))
        f_list = [f[:-2] for f in os.listdir(folder) if f.endswith(".p")]
            
        fname, ok = qtw.QInputDialog.getItem(
            self.loading_widget, "Choose Data Set", "Data Set:", f_list, 0, 
            False)
        
        if ok ==False:
            return
        
        try:
            pickle_file = open(os.path.join(folder, (fname + ".p")), "rb")
            self.name = pickle.load(pickle_file)
            self.directory_path = pickle.load(pickle_file)
            self.description_path = pickle.load(pickle_file)
            self.matrix_directory = pickle.load(pickle_file)
            self.matrices = pickle.load(pickle_file)
            self.records = pickle.load(pickle_file)
            self.groupings = pickle.load(pickle_file)
            self.data_types = pickle.load(pickle_file)
        except IOError:
            self.gb.loading_error("data set")
            return
        
        self.create_data_loading_widget()
        
    
    def symmetrize_matrix(self, matrix, method):
        """ Symmetrizes a matrix according to the given method.
        
        Parameters
        ----------
        matrix : list of list of float
            The matrix to symmetrize.
        method : int
            The method by which to symmetrize the matrix. Given that 
            matrix[i][j]=A and matrix[j][i]=B:
            method = 0 : matrix[i][j] = matrix[j][i] = max(A, B)
            method = 1 : matrix[i][j] = matrix[j][i] = min(A, B)
            method = 2 : matrix[i][j] = matrix[j][i] = mean(A, B)
            method = 3 : matrix[i][j] = matrix[j][i] = A
            method = 4 : matrix[i][j] = matrix[j][i] = B
        
        Returns
        -------
        matrix : list of list of float
            The symmetrized matrix.
        
        """
        
        for rownum, row in enumerate(matrix):
            for colnum in range(rownum):
                item = row[colnum]
                sym_item = matrix[colnum][rownum]
                
                if method == 0:
                    item = max(item, sym_item)
                elif method == 1:
                    item = min(item, sym_item)
                elif method == 2:
                    item = .5*(item + sym_item)
                elif method == 4:
                    item = sym_item
                
                row[colnum] = item
                matrix[colnum][rownum] = item
        
        return matrix
    
    
    def get_match_scores(self):
        
        """Stores match/nonmatch scores and ground truth values for each matrix.
        
        match_nonmatch_scores stores lists of all existing match scores, 
        nonmatch scores, all scores, and ground truth values for each matrix. 
        
        
        Parameters
        ----------
        None
        
        Returns
        -------
        bool:
            True if loaded successfully, and False otherwise.
        
        See Also
        --------
        VisualMetricAnalyzer class: more details on `match_nonmatch_scores`.
        """
        
        if len(self.matrices)==0:
            return True
        
        self.match_nonmatch_scores = {}
        for name in self.matrices:
            self.match_nonmatch_scores[name] = {
                "matches": [], "nonmatches": [], "all" : [], "gt": []}
           
        # Makes binary array such that match = 0; nonmatch = 1
        for row, record_1 in enumerate(self.records):
            for col, record_2 in enumerate(self.records):

                match_type = "nonmatches"
                if record_2.id in record_1.matches:
                    match_type = "matches"                   
                        
                for matrix_name in self.matrices:
                    matrix_info = self.matrices[matrix_name]
                    
                    if matrix_info["type"] == "Similarity":
                        ground_truth_options = ["nonmatches", "matches"]
                    else:
                        ground_truth_options = ["matches", "nonmatches"]
                    
                    try:
                        score = matrix_info["matrix"][row][col]
                    except (IndexError, ValueError):
                        self.gb.general_msgbox("Error", 
                            "Error loading scores; please try again.")
                        return False
                    
                    if score is not None and not np.isnan(score):               # NEW: removed the >=0 condition
                        self.match_nonmatch_scores[
                            matrix_name][match_type].append(
                            (score, record_1.id, record_2.id))
                        self.match_nonmatch_scores[matrix_name]["all"].append(
                            (score, record_1.id, record_2.id))
                        self.match_nonmatch_scores[matrix_name]["gt"].append(
                            ground_truth_options.index(match_type))
                       
        return True
    
###############################################################################
# Matrix generation methods
    def make_matrices(self, gen_dialog):
        """ Displays file type and metric choices for making distance matrices.
        
        The available image comparison methods are MSE (mean squared error), 
        NRMSE (normalized root mean squared error), SSIM (structural similarity 
        index), Euclidean (Euclidean/Frobenius distance between pixel values), 
        and Hamming (Hamming distance between pixel values). MSE, NRMSE, and 
        SSIM use the scikit-learn implementation.
        
        Parameters
        ----------
        mtx_widget: QWidget 
            The widget on which to display the options.
        
        Returns
        -------
        None
        
        See Also
        --------
        generate_matrices: Generates distance matrices for the chosen metrics 
            and data types.
            
        """
        
        
        self.selected_metrics = []
        self.selected_dtypes = []
        self.selected_fnames = []
        
        # Makes GUI for choosing what to generate
        gen_widget = qtw.QStackedWidget()
        gen_layout = qtw.QVBoxLayout()
        gen_dialog.setLayout(gen_layout)
        gen_layout.addWidget(gen_widget)   
        
        gen_dialog.setWindowTitle("Generate Matrices")
        gen_dialog.setWindowIcon(self.icon)
        mtx_layout = qtw.QGridLayout()

        font = qtgui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        
        title_label = self.gb.make_widget(
            qtw.QLabel("Matrix Generation Options"), font=font)
        
        title_label.setFixedHeight(50)
        mtx_layout.addWidget(title_label, 0, 0, 1, -1)
        mtx_layout.setSpacing(10)

        # Lists metrics to use for comparisons
        metrics_box=qtw.QGroupBox("Metrics")
        
        metrics_layout=qtw.QVBoxLayout()
        self.metrics = {"MSE": compare_mse, "NRMSE": compare_nrmse, 
                     "SSIM": compare_ssim, "Euclidean": None, "Hamming": None}                                      # , "Euclidean": np.linalg.norm, "Hamming": "h"
        for item in self.metrics:
            metrics_layout.addWidget(self.gb.make_widget(
                qtw.QCheckBox(gen_widget), "stateChanged", 
                partial(self.box_checked, "metric"), item))
            
        metrics_box.setLayout(metrics_layout)
        mtx_layout.addWidget(metrics_box, 1, 0)
        
        # List file types to use for comparisons
        dtypes_box=qtw.QGroupBox("File types to use")
        dtypes_layout=qtw.QVBoxLayout()
        for item in self.data_types:
            dtypes_layout.addWidget(self.gb.make_widget(qtw.QCheckBox(gen_widget), 
                                    "stateChanged", partial(self.box_checked, 
                                    "data type"), item))  
        dtypes_box.setLayout(dtypes_layout)
        mtx_layout.addWidget(dtypes_box, 1, 1)
        
        self.save_files_box = qtw.QCheckBox("Save as Text Files", gen_widget)
        mtx_layout.addWidget(self.save_files_box, 2, 0)
        mtx_layout.addWidget(self.gb.make_widget(qtw.QPushButton("Next"), 
                             "clicked", partial(self.name_matrices, 
                             gen_dialog, gen_widget)), 2, 1)
        
        
        font = qtgui.QFont()
        font.setPointSize(11)
        font.setBold(True)        
                
        mtx_choice_widget = qtw.QWidget()
        mtx_choice_widget.setLayout(mtx_layout)        
        mtx_naming_widget = qtw.QWidget()
        self.mtx_naming_layout = qtw.QVBoxLayout()
        mtx_naming_widget.setLayout(self.mtx_naming_layout)
        gen_widget.addWidget(mtx_choice_widget)
        gen_widget.addWidget(mtx_naming_widget)
        return gen_dialog.exec_()

    def box_checked(self, box_type, box):
        """ Updates the selected metrics or file types when a box is checked.
        
        Parameters
        ----------
        box_type: {"metric", "data type"}
            The type of box checked.
        box: QCheckBox
            The box checked.
        
        Returns
        -------
        None
        
        See Also
        --------
        make_matrices: Determines the types of distance matrices to generate.
        """
        
        
        if box.isChecked():
            if box_type=="metric":
                self.selected_metrics.append(str(box.text()))
            elif box_type == "svc":
                self.selected_svc = str(box.text())
            elif box_type == "matrix":
                self.selected_fusion_matrices.append(str(box.text()))
            else:
                self.selected_dtypes.append(str(box.text()))
        else:
            if box_type=="metric":
                self.selected_metrics.remove(str(box.text()))
            elif box_type == "matrix":
                self.selected_fusion_matrices.remove(str(box.text()))
            else:
                self.selected_dtypes.remove(str(box.text()))

    
    def name_matrices(self, mtx_dialog, mtx_widget, button):
        """ Gets the names and save locations of generated matrices.
        
        Parameters
        ----------
        mtx_widget: QWidget
            The widget that displays save options.
        button: QButton
            The button that triggered naming and saving the matrices.
            
        Returns
        -------
        None
        
        """
        
        font = qtgui.QFont()
        font.setPointSize(11)
        font.setBold(True)

        naming_title_label = self.gb.make_widget(qtw.QLabel("Matrix Names"), 
                                          font=font)
        
        naming_title_label.setFixedHeight(50)
        self.mtx_naming_layout.addWidget(naming_title_label)
        
        for metric in self.selected_metrics:
            for dtype in self.selected_dtypes:
                name = dtype + "_" + metric + "_matrix"
                edit = qtw.QLineEdit(name)
                edit.setReadOnly(False)
                self.mtx_naming_layout.addWidget(edit)
                self.selected_fnames.append(edit)     
                
        # Lets user choose where to save (defaults to _generated_matrices)
        if self.matrix_directory == "":
            self.matrix_directory = os.path.join(self.directory_path, 
                                                 "_generated_matrices")
        
        self.mtx_naming_layout.addWidget(qtw.QLabel("Save in"))
        save_layout = qtw.QHBoxLayout()
        self.matrix_dir_edit = qtw.QLineEdit(self.matrix_directory)
        save_layout.addWidget(self.matrix_dir_edit)        
        save_layout.addWidget(self.gb.make_widget(
            qtw.QPushButton("Change"), "clicked", self.change_save_loc))
        
        self.mtx_naming_layout.addLayout(save_layout)
        self.mtx_naming_layout.addWidget(self.gb.make_widget(
            qtw.QPushButton("Done"), "clicked", partial(self.generate_matrices, mtx_dialog)))
            
        mtx_widget.setCurrentIndex(1)
    
        
    def change_save_loc(self):
        """ Changes where to save generated matrices.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        """
        
        try:
            original_directory = qtw.QFileDialog.getExistingDirectory(
                None, "Choose Matrix Directory", self.matrix_dir_edit.text())
            matrix_directory = self.fix_path(str(original_directory))
            
            if matrix_directory == "":
                return
            
            # Warns user if save location could cause issues
            if matrix_directory in self.directory_path:
                reply = self.gb.general_msgbox(
                        "Warning", ("Have you ensured that matrices were not "
                        "saved inside directories that contain data files?"), 
                        cancel=True)
                if reply == 1:
                    return
        except IOError:
            self.gb.loading_error("directory")
            return
        
        self.matrix_dir_edit.setText(matrix_directory)

    
    def generate_matrices(self, gen_dialog):
        """ Generates the selected distance matrices for the data set.
        
        Creates the distance matrices corresponding to the file types in 
        `selected_dtypes` and metrics in `selected_metrics`. Adds these 
        matrices to the data set and, if chosen, saves the matrices as text 
        files as well.
        
        Parameters
        ----------
        mtx_widget QWidget
            The widget on which matrix generation options were displayed.
        
        Returns
        -------
        None
        
        See Also
        --------
        make_matrices: Determines the types of distance matrices to generate.

        """
    
        if len(self.selected_fnames) != 0:
            self.matrix_directory = self.matrix_dir_edit.text()
            main_path, new_dir = os.path.split(self.matrix_directory)
            try:
                os.makedirs(self.matrix_directory, exist_ok=True)
            except IOError:
                self.gb.loading_error("matrix directory")
                return

        fname_index=0
        for findex, ftype in enumerate(self.selected_dtypes):
            for mindex, metric in enumerate(self.selected_metrics):
                
                new_matrix = []

                fname = self.selected_fnames[fname_index].text()
                fname_index+=1
                
                # Makes progress widget
                progress_dialog = qtw.QProgressDialog(
                    "Generating matrix " + fname + "...", "Cancel", 0, 
                    len(self.records))
                progress_dialog.setWindowModality(qtcore.Qt.WindowModal)
                progress_dialog.setAutoClose(True)
                progress_dialog.resize(500,100)
                progress_dialog.setWindowIcon(self.icon)
                progress_dialog.setWindowTitle("VEMOS")
                progress_dialog.setAutoClose(True)
                progress_dialog.show() 
                progress_dialog.setValue(1)
                    
                new_matrix = np.zeros((len(self.records), len(self.records)))
                new_matrix.fill(np.nan)
                
                for index, record in enumerate(self.records):
                    
                    progress_dialog.setValue(index)
                    if progress_dialog.wasCanceled():
                        return
                    
                    for index2, record2 in enumerate(self.records):                            # Only need to go thru 1/2 of records--fix after fixing symmetrization

                        if ftype in record.files and ftype in record2.files:
                            try:
                                path1 = os.path.join(self.directory_path, 
                                                     record.files[ftype])
                                path2 = os.path.join(self.directory_path, 
                                                     record2.files[ftype])
                                im1 = mpl.image.imread(path1)
                                im2 = mpl.image.imread(path2)
                            except (IOError, SyntaxError) as e:
                                self.gb.general_msgbox(
                                    "Matrix Generation Error", ("Image "
                                    "similarity matrices can only be generated"
                                    " from images or segmentations. Please "
                                    "select a different data type."))
                                return False
                        
                            if len(im1.shape) == 3: 
                                im1 = im1[:,:,0]
                                im2 = im2[:,:,0]
                                
                            # Crop larger image if necessary
                            size1 = np.array(im1.shape)
                            size2 = np.array(im2.shape)
                            
                            cropx = min(size1[0], size2[0])
                            cropy = min(size1[1], size2[1])
                                
                            y,x = im1.shape
                            startx = x//2-(cropx//2)
                            starty = y//2-(cropy//2)
                            im1 = im1[starty:starty+cropy,startx:startx+cropx]          # I think something's wrong w/ the cropping--try imshowing the imgs after cropping    
                               
                            y,x = im2.shape
                            startx = x//2-(cropx//2)
                            starty = y//2-(cropy//2)
                            im2 = im2[starty:starty+cropy,startx:startx+cropx]
                
                            try:
                                if metric=="Euclidean":
                                    new_matrix[index][index2] = np.linalg.norm(
                                        im1-im2)    # check
                                elif metric=="Hamming":
                                    new_matrix[index][index2]=(im1!=im2).sum()
                                else:
                                    method = self.metrics[metric]
                                    new_matrix[index][index2]=method(im1, im2)
                    
                            except ValueError:
                                self.gb.general_msgbox(
                                    "Matrix Generation Error", ("Score "
                                    "generation failed; please try again."))
                                return
                
                # Add matrix to self.matrices and save files if required
                if len(self.selected_fnames) != 0:
                    save_address = os.path.join(self.matrix_directory, 
                                                fname + ".txt")
                    np.savetxt(save_address, new_matrix)
                else:
                    save_address=""
                
                mtype = "dissimilarity"
                if metric == "SSIM":
                    mtype = "Similarity"
                self.matrices[fname] = {"matrix": new_matrix, 
                                        "path": save_address, 
                                        "type": mtype, "format": "Matrix"}
        
        self.get_match_scores()        
        gen_dialog.accept()
        gen_dialog.close()
        self.update = True
        return
        
    
    def make_fused_matrices(self, fusion_dialog):                                    # nearly the same as make_matrices--combine?
        """ Displays SVC and matrix choices for fusing matrices.
        
        Matrices can be fused using linear, polynomial, or radial basis 
        function (RBF) support vector machines.
        
        Parameters
        ----------
        fusion_dialog: QDialog 
            The dialog on which to display the options.
        
        Returns
        -------
        None
        
        See Also
        --------
        fuse_matrices: Fuses the chosen matrices using the selected method. 
        
        """
        
        # Choose matrices to fuse

        self.selected_fusion_matrices = []
        self.selected_fnames = []
        self.selected_svc = ""
        
        fusion_widget = qtw.QStackedWidget()
        fusion_layout = qtw.QVBoxLayout()
        fusion_dialog.setLayout(fusion_layout)
        fusion_layout.addWidget(fusion_widget)        
        
        # Makes GUI for choosing what matrices to fuse and how to fuse them
        fusion_dialog.setWindowTitle("Fuse Matrices")
        fusion_dialog.setWindowIcon(self.icon)
        mtx_layout = qtw.QGridLayout()

        font = qtgui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        
        title_label = self.gb.make_widget(
            qtw.QLabel("Matrix Fusion Options"), font=font)
        
        title_label.setFixedHeight(50)
        mtx_layout.addWidget(title_label, 0, 0, 1, -1)
        mtx_layout.setSpacing(10)

        # Lists SVC types to use
        svcs_box=qtw.QGroupBox("SVC Type")
        
        svcs_layout=qtw.QVBoxLayout()
        self.svcs = {"Linear": svm.SVC(kernel='linear'), 
                     "RBF": svm.SVC(kernel='rbf', gamma=0.7), 
                     "Polynomial": svm.SVC(kernel='poly', degree=3, 
                                           gamma='auto')}
        for item in self.svcs:
            svcs_layout.addWidget(self.gb.make_widget(
                qtw.QRadioButton(fusion_widget), "toggled", 
                partial(self.box_checked, "svc"), item))
            
        svcs_box.setLayout(svcs_layout)
        mtx_layout.addWidget(svcs_box, 1, 0)
        
        # Lists file types to use for comparisons
        matrices_box=qtw.QGroupBox("Matrices to use")
        matrices_layout=qtw.QVBoxLayout()
        for item in self.matrices:
            matrices_layout.addWidget(self.gb.make_widget(
                qtw.QCheckBox(fusion_widget), "stateChanged", 
                partial(self.box_checked, "matrix"), item))  
        matrices_box.setLayout(matrices_layout)
        mtx_layout.addWidget(matrices_box, 1, 1)
        
        self.save_file_box = qtw.QCheckBox("Save as Text File", fusion_widget)
        mtx_layout.addWidget(self.save_file_box, 2, 0)
        mtx_layout.addWidget(self.gb.make_widget(qtw.QPushButton("Next"), 
                             "clicked", partial(self.name_fused_matrices, 
                             fusion_dialog, fusion_widget)), 2, 1)
        
        
        font = qtgui.QFont()
        font.setPointSize(11)
        font.setBold(True)        
                
        mtx_choice_widget = qtw.QWidget()
        mtx_choice_widget.setLayout(mtx_layout)        
        mtx_naming_widget = qtw.QWidget()
        self.mtx_naming_layout = qtw.QVBoxLayout()
        mtx_naming_widget.setLayout(self.mtx_naming_layout)
        fusion_widget.addWidget(mtx_choice_widget)
        fusion_widget.addWidget(mtx_naming_widget)
        return fusion_dialog.exec_()

    
    def name_fused_matrices(self, fusion_dialog, mtx_widget, button):
        """ Gets the names and save locations of fused matrices.
        
        Parameters
        ----------
        mtx_widget: QWidget
            The widget that displays save options.
        button: QButton
            The button that triggered naming and saving the matrices.
            
        Returns
        -------
        None
        
        """
        
        self.selected_fnames = []

        font = qtgui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        
        
        naming_title_label = self.gb.make_widget(qtw.QLabel("Matrix Names"), 
                                          font=font)
        
        naming_title_label.setFixedHeight(50)
        self.mtx_naming_layout.addWidget(naming_title_label)
        
        name = self.selected_svc + "_fused_matrix"

        edit = qtw.QLineEdit(name)
        edit.setReadOnly(False)
        self.mtx_naming_layout.addWidget(edit)
        self.selected_fnames.append(edit)
                
                

        # Lets user choose where to save (defaults to _generated_matrices)
        if self.matrix_directory == "":
            self.matrix_directory = os.path.join(self.directory_path, 
                                                 "_generated_matrices")
        
        self.mtx_naming_layout.addWidget(qtw.QLabel("Save in"))
        save_layout = qtw.QHBoxLayout()
        self.matrix_dir_edit = qtw.QLineEdit(self.matrix_directory)
        save_layout.addWidget(self.matrix_dir_edit)        
        save_layout.addWidget(self.gb.make_widget(
            qtw.QPushButton("Change"), "clicked", self.change_save_loc))
        
        self.mtx_naming_layout.addLayout(save_layout)
        self.mtx_naming_layout.addWidget(self.gb.make_widget(
            qtw.QPushButton("Done"), "clicked", partial(self.fuse_matrices, 
            fusion_dialog)))
            
        mtx_widget.setCurrentIndex(1)
    
    def fuse_matrices(self, fusion_dialog):
        """ Fuses the chosen matrices using the selected SVC type.
        
        Fuses the chosen distance matrices according to `selected_svc`, Adds 
        this matrix to the data set and, if chosen, saves the matrix as a
        text file as well.
        
        Parameters
        ----------
        mtx_widget QWidget
            The widget on which matrix generation options were displayed.
        Returns
        -------
        None
        
        See Also
        --------
        make_matrices: Determines the types of distance matrices to generate.
        
        """

        # Create feature matrix from selected (dis)similarity matrices
        X = []
        for mname in self.selected_fusion_matrices:
            X.append(np.nan_to_num(self.matrices[mname]["matrix"]).flatten())
        
        # Gets ground truth values
        type0 = self.matrices[self.selected_fusion_matrices[0]]["type"]
        for matrix in self.selected_fusion_matrices[1:]:
            if self.matrices[matrix]["type"] != type0:
                self.gb.general_msgbox(
                    "Matrix Fusion Error", ("Similarity matrices cannot be"
                    " fused with dissimilarity matrices. Please try again."))
                return
        
        gt = self.match_nonmatch_scores[self.selected_fusion_matrices[0]]["gt"]
        
        # Adds only pairs with IDs to X
        X = []
        X_ids = []
        X_gt = []
        
        all_scores_0 = self.match_nonmatch_scores[
            self.selected_fusion_matrices[0]]["all"]
        
        for index, item in enumerate(all_scores_0):
            scores = [item[0]]
            
            for other_matrix in self.selected_fusion_matrices[1:]:
                other_items = self.match_nonmatch_scores[other_matrix]["all"]
                for other_item in other_items:
                    if item[1:] == other_item[1:]:
                        scores.append(other_item[0])
                        break
                else:
                    break
            else:
                X.append(scores)
                X_ids.append(item[1:])
                X_gt.append(gt[index])
        
        
        random.seed(57)                                  #seeded for submission
        X = np.array(X)
        y = X_gt
        clf = self.svcs[self.selected_svc]
        clf.fit(X, y)
        D = clf.decision_function(X)

        # Un-flattens new scores into matrix
        dim = len(self.records)
        d_matrix = np.zeros((dim, dim))
        d_matrix.fill(np.nan)
        
        # Removes negative values
        D = D + abs(np.amin(D))
        for index, score in enumerate(D):
            ids = X_ids[index]
            index1 = self.index_from_id(ids[0])
            index2 = self.index_from_id(ids[1])
            
            d_matrix[index1][index2] = score
            d_matrix[index2][index1] = score

        fname = self.selected_fnames[0].text()
        
        if len(self.selected_fnames) != 0:
            save_address = os.path.join(self.matrix_directory, fname + ".txt")
            np.savetxt((save_address), d_matrix)
             
        # Adds new matrix to self.matrices and self.match_nonmatch_scores
        self.matrices[fname] = {"matrix": d_matrix, "path": save_address, 
                                 "type": type0, "format": "Matrix"}
        self.get_match_scores()

       
#        fig = plt.figure()
#        ax = plt.gca()
#        #plt.subplots_adjust(wspace=0.4, hspace=0.4)
#        
#        # Only works for 2 matrices
#        X0, X1 = X[:, 0], X[:, 1]       
#        x_min, x_max = X0.min() - 1, X0.max() + 1
#        y_min, y_max = X1.min() - 1, X1.max() + 1
#        xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
#                             np.arange(y_min, y_max, .02))
#        
#        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#        Z = Z.reshape(xx.shape)
#        out = ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
#        
#        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
#        ax.set_xlim(xx.min(), xx.max())
#        ax.set_ylim(yy.min(), yy.max())
#        ax.set_xlabel('Score 1')
#        ax.set_ylabel('Score 2')          
#        plt.show()
        
        self.update = True
        fusion_dialog.accept()
        fusion_dialog.close()
        gc.collect()
        return
                    
###############################################################################

    def fix_path(self, path):
        """ Fixes formatting differences in some file paths.
        
        Description files created in Windows may use different slashes in file 
        paths than description files created on other operating systems; this 
        method ensures that all paths use the same format.
        
        Parameters
        ----------
        path : str
            The file path to be corrected.
        
        Returns
        -------
        path : str
            The corrected file path.
        """ 
    
        # Fixes the path for Windows operating systems only
        
        if sys.platform.lower().startswith('win'):
            
            parts = re.split(r'\\|/', path)
            path = parts[0]
            del parts[0]
    
            for part in parts:
                path = path + "\\" + part
    
        return path
    
    
    def get_file_path(self, record, data_type):
        """ Returns the complete path to the file for a record and data type.
        
        Parameters
        ----------
        record : DataRecord
            The record from which to retrieve the file path.
        data_type : str
            The data type for which to retrieve the file path.
        
        Returns
        -------
        str
            The full path to the requested data file for the given record.
            
        """ 
        
        return os.path.join(self.directory_path, record.files[data_type])
    
    
    def record_from_id(self, id_num):
        """ Given an ID number, returns the corresponding DataRecord.
        
        Parameters
        ----------
        id_num : str
            The ID of the record to retrieve.
        
        Returns
        -------
        record : DataRecord
            The record corresponding to the parameter ID, or -1 if the record 
            was not found.
            
        """ 
        
        for record in self.records:
            if record.id == id_num:
                return record
        
        return -1
    

    def index_from_id(self, id_num):
        """ Given an ID, returns the corresponding index in the record list.
        
        Parameters
        ----------
        id_num : str
            The ID of the record to retrieve.
        
        Returns
        -------
        index : int
            The index of the record with the parameter ID, or -1 if the record 
            was not found.
            
        """ 
        
        for index in range(len(self.records)):
            if self.records[index].id == id_num:
                return index
                
        return -1
                
###############################################################################
# Data output methods
                
    def print_all_data(self): 
        """ Prints the data set's records, score matrices, and groupings.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """                                                   
        
        print("--- PRINTING ALL DATA ---")
            

        print("Data Records:\n")
        for cur_record in self.records:
            print(cur_record.id)
            cur_record.print_data()
        
        print("\nMatrices:")
        for matrix_name in self.matrices:
            print("Name: ", matrix_name)
            print("Path: ", self.matrices[matrix_name]["path"])
            print("Matrix Type:", 
                  self.matrices[matrix_name]["type"])
            print("Scores: ", self.matrices[matrix_name]["matrix"])
                
            print("\nGroupings:")
            for grouping_name in self.groupings:
                print("Grouping: ", grouping_name)
                for group_name in self.groupings[grouping_name]:
                    print(group_name, "IDs: ")
                    print(self.groupings[grouping_name][group_name])
            
            print("\nData Types:")
            for type_name in self.data_types:
                print("Name: ", type_name)
                print("Type: ", self.data_types[type_name]["type"])
                print("Format: ", self.data_types[type_name]["format"])
                print("Extension: ", self.data_types[type_name]["extension"])

        
    def save_current_description_file(self):
        """ Saves the data description in the current working file.   
        
        Parameters
        ----------
        None
        
        Returns
        -------
        bool
            Whether or not the file was saved successfully.
        
        See Also
        --------
        save_description_file_as : Requests where to save the description file.
        save_description_file : Saves the description to the parameter file.
        """

        if self.description_path != "":
            return self.save_description_file(self.description_path)
        else:
            return self.save_description_file_as()
    

    def save_description_file_as(self):
        """ Saves a data description in the location of the user's choice.  
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        See Also
        --------
        save_current_description_file : Saves the description in the current 
            description file.
        save_description_file : Saves the description to the parameter file.
        """

        # Asks user where to save
        try:
            file_name = qtw.QFileDialog.getSaveFileName(
                caption='Save Data Description As', 
                directory=self.directory_path, 
                filter='*.txt')
            if file_name is None or file_name[0]=="":
                return False
            file_name=str(file_name[0])
        except IOError:
            self.gb.general_msgbox("Error Saving Description File", 
                                       "Please try again.")
            return False
        
        
        save_success = self.save_description_file(file_name)
        self.description_path = file_name
            
        return save_success
        
     
    def save_description_file(self, fname):
        """ Saves the data description to the given file.
        
        Parameters
        ----------
        fname : str
            The name of the file in which to save the description.
        
        Returns
        -------
        bool
            True if successful, False otherwise.
        
        See Also
        --------
        save_current_description_file : Saves the description in the current 
            description file.
        save_description_file_as : Requests where to save the description file.
        """
        
        if fname is not None:
            try:
                with open(fname, 'w') as file:
                    for record in self.records:
                        file.writelines(record.get_description_file_format())
                return True
            except IOError:
                self.gb.general_msgbox("Error Saving Description File", 
                                       "Please try again.")
                return False

                   
    def open_description_file(self):
        """ Opens the description file as a text file. 
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
        if self.description_path or self.save_description_file_as():
            os.startfile(self.description_path)
            

    def save(self, window, ask=True):
        """ Pickles the current data set.
        
        Parameters
        ----------
        window: QMainWindow
            The parent window of the QDialog that asks about saving.
        ask: {True, False}
            Stores if the user should be asked before saving the data set. 
        
        Returns
        -------
        None
        """
        
        if ask:
            result = self.gb.general_msgbox(
                "Save Before Closing?", ("Save this data set before closing?"), 
                no=True, cancel=True) 
            
            if result == 1:
                return True
            elif result == 2:
                return False       
        
        item, ok = qtw.QInputDialog.getText(window, "Save As", "Name:",
                                            qtw.QLineEdit.Normal, self.name)
        if ok == False:
            return False
        
        if (item + ".p") in os.listdir():
            result = self.gb.general_msgbox(
                "Replace Data Set", ("There is already a data set with this "
                "name. Would you like to replace it?"), no=True, cancel=True)
            if result == 1:
                return self.close(window, ask = False)
            elif result == 2:
                return False
        
        try:
            self.name = item

            folder = os.path.dirname(os.path.realpath(__file__))
            save_file = open(os.path.join(folder, self.name + ".p"), "wb")
            
            pickle.dump(self.name, save_file)
            pickle.dump(self.directory_path, save_file)
            pickle.dump(self.description_path, save_file)
            pickle.dump(self.matrix_directory, save_file)
            pickle.dump(self.matrices, save_file)
            pickle.dump(self.records, save_file)
            pickle.dump(self.groupings, save_file)
            pickle.dump(self.data_types, save_file)
            pickle.dump(self.has_files, save_file)

            save_file.close()
            return True
        
        except (IOError, PermissionError) as e:
            self.gb.general_msgbox("Save Error", "Unable to save files.")
            return False

    
    def cancel(self, event):
        """ Closes the loading widget when the user clicks "Cancel".
        
        Exits the application as well if the loading widget is the only widget 
        open.
        
        Parameters
        ----------
        event: QEvent
            The event that triggered canceling the dialog.
        
        Returns
        -------
        None
        """
        
        if self.update:
            self.update = False
            self.loading_widget.accept()
            self.loading_widget.close()
        else:
            self.exit_application()
        
class LoadingWidget(qtw.QDialog):

    def __init__(self, parent=None):
        super(LoadingWidget, self).__init__(parent)
        
        self.setWindowFlag(qtcore.Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(qtcore.Qt.WindowContextHelpButtonHint, False)         
