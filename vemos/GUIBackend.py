# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 13:50:58 2018

@author: Eve Fleisig

This module provides methods for setting up the Visual Metric Analyzer and 
Visual Record Browser GUIs.
"""

import os
import gc
from functools import partial
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as 
    FigureCanvas)
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as 
    NavigationToolbar)

import PyQt5.QtGui as qtgui
import PyQt5.QtWidgets as qtw
import PyQt5.QtCore as qtcore
    
class GUIBackend(object):
    """Provides methods for setting up the visualization GUIs.
    
    Includes error messages, methods for setting up PyQt components, and 
    methods for reading in different file types.
    
    Parameters
    ----------
    None
    
    Attributes
    ----------
    None
    
    """
    
    def make_widget(self, widget_type, to_connect=None, connecting_method=None,
                    text=None, pos=None, resizing=None, value=None, 
                    minimum=None, maximum=None, font=None, window_title=None, 
                    items=None, index=None, checked=None, add_to=None, 
                    is_partial=False, editable=None):
        """Makes a custom widget with the parameter characteristics.
        
        Parameters
        ----------
        widget_type : QWidget
            The type of widget to create.
        to_connect : function, optional
            The signal to connect `connecting_method` to (e.g., "stateChanged")
        connecting_method : function, optional
            The method to which the signal should be connected.
        text : str, optional
            The text to display on the widget.
        pos : tuple of int, optional
            The (x, y) position to which the widget should be moved.
        resizing : tuple of int, optional
            The (w, h) size of the widget.
        value : str or int or float, optional
            The initial value of the widget.
        minimum : float or int, optional
            The minimum permissible value for the widget
        maximum : float or int, optional
            The minimum permissible value for the widget
        font : QFont, optional
            The font of the widget's text.
        window_title : str, optional
            If the widget is a window, gives the title it should have.
        items : list of str, optional
            The options the widget should have.
        index : int, optional
            The initial index of a widget with multiple options.
        checked : bool, optional
            If the widget is a check box, stores whether it should be checked.
        add_to : QWidget, optional
            The larger widget to which this widget should be added.
        is_partial : bool, optional
            Whether to use partial() to include more information when 
            connecting the widget to a signal.
        editable : bool, optional
            Whether the widget should be editable.
        
        Returns
        -------
        widget : QWidget
            The widget created.
            
        """
        
        widget = widget_type
        
        if items:
            for item in items:
                widget.addItem(item)
        
        if to_connect != None:
            
            if is_partial:
                if items:
                    connecting_method = partial(
                        connecting_method, add_to, values=items)
                else:
                    connecting_method = partial(connecting_method, add_to)                
                
            if to_connect == "triggered":
                widget.triggered.connect(connecting_method)
                
            elif to_connect == "stateChanged":
                widget.stateChanged.connect(lambda:connecting_method(widget))
            
            elif to_connect == "toggled":
                widget.toggled.connect(lambda:connecting_method(widget))
                    
            elif to_connect == "itemSelectionChanged":
                widget.itemSelectionChanged.connect(connecting_method)
                
            elif to_connect == "clicked":
                widget.clicked.connect(connecting_method)
            
            elif to_connect =="itemClicked":
                widget.itemClicked.connect(connecting_method)
                
            elif to_connect == "valueChanged":
                widget.valueChanged.connect(connecting_method)
                
            elif to_connect == "currentIndexChanged":
                widget.currentIndexChanged.connect(connecting_method)
            
        if text != None:
            widget.setText(text)
            
        if pos != None:
            widget.move(*pos)
            
        if resizing != None:
            widget.resize(*resizing)
            
        if value != None:
            widget.setValue(value)
            
        if minimum != None:
            widget.setMinimum(minimum)
            
        if maximum != None:
            widget.setMaximum(maximum)
            
        if font != None:
            widget.setFont(font)
            
        if window_title != None:
            widget.setWindowTitle(window_title)
        
        if index != None:
            widget.setCurrentIndex(index)
            
        if checked != None:
            widget.setChecked(checked)
                
        if add_to != None:
            add_to.append(widget)
        
        if editable != None:
            widget.setEditable(editable)
   
        return widget
        
###############################################################################
# Message box methods        
    def general_msgbox(self, title, text, info="", 
                       icon=qtw.QMessageBox.Warning, no=False, cancel=False):
        """Displays a custom message box.
        
        By default, only an "OK" button is displayed. If `cancel` is true, 
        displays "OK" and "Cancel" buttons. If `no` is also true, displays 
        "Yes", "No", and "Cancel" buttons. 
        
        Parameters
        ----------
        title : str
            The title of the message box.
        text : str
            The text of the message box.
        info : str, optional
            Additional text to display in the message box.
        icon : QMessageBox.Icon
            The icon to display in the message box.
        no : {False, True}
            Whether to include a yes/no button.
        cancel : {False, true}
            Whether to include a cancel button.
        Returns
        -------
        widget : QWidget
            The widget created.
        """
        
        msg = qtw.QMessageBox()
        msg.setIcon(icon)
        msg.setText(text)
        msg.setInformativeText(info)
        msg.setWindowTitle(title)
        
        if cancel and no:
            msg.addButton(qtw.QPushButton('Yes'), qtw.QMessageBox.YesRole)
            msg.addButton(qtw.QPushButton('No'), qtw.QMessageBox.NoRole)
            msg.addButton(qtw.QPushButton('Cancel'), 
                          qtw.QMessageBox.RejectRole)
        elif cancel:
            msg.addButton(qtw.QPushButton('OK'), qtw.QMessageBox.YesRole)
            msg.addButton(qtw.QPushButton('Cancel'), 
                          qtw.QMessageBox.RejectRole)

        return msg.exec_()
                            

    def no_description_error(self):
        """Error message if trying to use a feature that requires data files.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        See Also
        --------
        general_msgbox: Displays a custom message box.
        
        """
        
        self.general_msgbox("Feature Unavailable", "This feature cannot be "
                            "displayed without loading data files.")
    
                        
    def loading_error(self, to_load, v2=False):
        """Displays an error message if data could not be loaded.
        
        Parameters
        ----------
        to_load : str
            The information that could not be loaded.
        v2 : {False, True}, optional
            Whether to use the second version of the error message.
    
        Returns
        -------
        None
        
        See Also
        --------
        general_msgbox: Displays a custom message box.
        """
        
        if v2:
            self.general_msgbox(
                "Loading Failed", "The " + to_load + " was corrupt or "
                "unavailable. Please try again.")
        else:
            self.general_msgbox(
                "Loading Failed", "Please select a valid " + to_load + ".")
    

    def no_selection_error(self):
        """Error if trying to open an analysis without selecting matrices.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        See Also
        --------
        general_msgbox: Displays a custom message box.
        """
        self.general_msgbox(
            "No Matrices Selected", ("Please check the boxes of the matrices "
            "you would like to use for this analysis."))
        
        
###############################################################################
# Image display methods
         
    def show_pic(self, axis, path1, path2=None, embed=False, gray=False):
        """Displays one or two images on the given axis.
        
        If two images are being displayed on the same axis, adds space around 
        the images to make them the same size.
        
        Parameters
        ----------
        axis : matplotlib.axes.Axes
            The axis on which to display the images.
        path1 : str
            The file path to the first image.
        path2 : str, optional
            The file path to the second image.
        embed : {False, True}
            Whether to embed the image in an existing figure.
        gray : {False, True}
            Whether to display the image in grayscale.
        
        Returns
        -------
        None
        
        See Also
        --------
        VisualMetricAnalyzer.view_selected_records: Displays images and other 
            file types in the Visual Metric Analyzer.
        DataRecordBrowser.show_record: Displays images and other file 
            types in the Data Record Browser.
        """
        
        axis.clear()

        if not embed:
            plt.ion()
        
        I = None
        
        if path1 != "Unavailable":
            try:
                I = mpl.image.imread(path1)
            except (IOError, SyntaxError) as e:
                self.loading_error("image/segmentation file", v2=True)
                path1 = "Unavailable"
        
        if path1 == "Unavailable":
            if path2 == None:
                axis.text(.5, .5, "File Unavailable", fontsize=14, 
                          bbox={'facecolor':'white', 'pad':10}, 
                          horizontalalignment='center',
                          verticalalignment='center', transform=axis.transAxes)
                return
            else:
                axis.text(.4, .5, "File Unavailable", fontsize=14, 
                          bbox={'facecolor':'white', 'pad':10}, 
                          horizontalalignment='right',
                          verticalalignment='center', transform=axis.transAxes)
                
        axis.set_aspect("equal")
        
            
        # If two pictures, displays both on one axis.
        if path2 != None: 
 
            if path2 != "Unavailable":
                try:
                    pic2 = mpl.image.imread(path2)
                except (IOError, SyntaxError) as e:
                    self.loading_error("image/segmentation file", v2=True)
                    path2 = "Unavailable"
            
            if path2 == "Unavailable":
                axis.text(.6, .5, "File Unavailable", fontsize=14, 
                          bbox={'facecolor':'white', 'pad':10}, 
                          horizontalalignment='left',
                          verticalalignment='center', transform=axis.transAxes)
                if path1 == "Unavailable":
                    pic2 = np.zeros((10, 10))
                    plt.gray()
                else:
                    pic2 = np.zeros(I.shape)
                    
            if I is None:
                pic1 = np.zeros(pic2.shape)
            else:
                pic1 = I    
           
            size1 = np.array(pic1.shape)
            size2 = np.array(pic2.shape)
                
            if pic1.ndim!=pic2.ndim:
                if pic1.ndim==2:
                    pic1 = np.concatenate(pic1, pic1, pic1)
                else:
                    pic2 = np.concatenate(pic2, pic2, pic2)
            
            # Adds space around images to give them the same size.
            max_size = np.maximum(size1, size2)
            if pic1.ndim == 2:
                new_size = (max_size[0], 2*max_size[1] + 10)
            elif pic1.ndim == 3:
                new_size = (max_size[0], 2*max_size[1] + 10, max_size[2])
            

            I = np.zeros(new_size)
            offset1 = ((max_size-size1)/2).astype(int)
            offset2 = ((max_size-size2)/2).astype(int)
            offset2[1] += max_size[1] + 10
            

            xlim1 = offset1[0] + size1[0]     
            xlim2 = offset2[0] + size2[0]
            ylim1 = offset1[1] + size1[1]
            ylim2 = offset2[1] + size2[1]
            
            if I.ndim == 2:
                I[offset1[0]:xlim1, offset1[1]:ylim1] = pic1
                I[offset2[0]:xlim2, offset2[1]:ylim2] = pic2
            elif I.ndim == 3:
                I[offset1[0]:xlim1, offset1[1]:ylim1,:] = pic1
                I[offset2[0]:xlim2, offset2[1]:ylim2,:] = pic2
            else:
                return
                
            I[offset1[0]:xlim1, offset1[1]:ylim1,...] = pic1
            I[offset2[0]:xlim2, offset2[1]:ylim2,...] = pic2

            if path1 != "Unavailable":
                I = I.astype(pic1.dtype)
            else:
                I = I.astype(pic2.dtype)

        if gray:
            axis.imshow(I, interpolation="none", cmap='gray')
        else:
            axis.imshow(I, interpolation="none")
        

    
    def show_curve(self, axis, path1, path2=None, embed=False, style=None, 
                   connect=True):
        """Displays one or two curves or point clouds on the given axis. 
        
        If two curves/point clouds are being displayed on the same axis, adds 
        space around them to make them the same size.
        
        Parameters
        ----------
        axis : matplotlib.axes.Axes
            The axis on which to display the curves.
        path1 : str
            The file path to the first curve.
        path2 : str, optional
            The file path to the second curve.
        embed : {False, True}
            Whether to embed the curve in an existing figure.
        style : str, optional
            The matplotlib style to use for plotting; defaults to 'r.-'.
        connect : {True, False}
            Whether to connect the plotted points with lines.
        
        Returns
        -------
        None
        
        See Also
        --------
        VisualMetricAnalyzer.view_selected_records: Displays curves and other 
            file types in the Visual Metric Analyzer.
        DataRecordBrowser.show_record: Displays curves and other file types in
            the Data Record Browser.
            
        """
            
        if embed:
            axis.clear()
        if not style:
            style = 'r.-'
        
        if not connect:
            style = style[:-1]
        

        style1 = style
        
        if path1 != "Unavailable":
            try:
                curve1 = np.loadtxt(path1).T
                curve1[0] -= np.amin(curve1[0])
            except (IOError, OSError, UnicodeDecodeError, SyntaxError, 
                    ValueError) as e:
                self.loading_error("curve file", v2=True)
                path1 = "Unavailable"
        
        if path1 == "Unavailable":
            curve1 = np.array([[0, 0, 300, 300], [0, 300, 300, 0]])
            style1 = "None"
            
            if path2 == None:
                axis.text(.5, .5, "File Unavailable", fontsize=14, 
                          bbox={'facecolor':'white', 'pad':10}, 
                          horizontalalignment='center',
                          verticalalignment='center', transform=axis.transAxes)
            else:
                axis.text(.4, .5, "File Unavailable", fontsize=14, 
                          bbox={'facecolor':'white', 'pad':10}, 
                          horizontalalignment='right',
                          verticalalignment='center', transform=axis.transAxes)

            
        axis.plot(curve1[1], 1 - curve1[0], style1, markersize=3)
        axis.margins(.05, .05)
        axis.set_aspect("equal")
            
        
        # If two curves, displays both on one axis.
        style2=style
        if path2 != None:
            
            if path2 != "Unavailable":
                try:
                    curve2 = np.loadtxt(path2).T
                    curve2[0] -= np.amin(curve2[0])
                
                    if path1 != "Unavailable":
                        curve2[1] += (np.amax(curve1[1])-np.amin(curve2[1]))+5   
                    else:
                        curve2[1] += (np.amax(curve2[1])) + 5 
                        
                except (IOError, OSError, UnicodeDecodeError, SyntaxError,
                        ValueError) as e:
                    self.loading_error("curve file", v2=True)
                    path2 = "Unavailable"
            
            if path2 == "Unavailable":
                style2="None"
                axis.text(.6, .5, "File Unavailable", fontsize=14, 
                          bbox={'facecolor':'white', 'pad':10}, 
                          horizontalalignment='left',
                          verticalalignment='center', transform=axis.transAxes)
                curve2 = np.array([[0, 0, 300, 300], [0, 300, 300, 0]])
                curve2[1] += int(np.amax(curve1[1]) - np.amin(curve2[1])) + 400 
                 

            axis.plot(curve2[1], 1 - curve2[0], style2)
        

    def show_text(self, path):
        """Returns a text edit containing the text from the file at `path`.
        
        Parameters
        ----------
        path : str
            The file path to the text.
        
        Returns
        -------
        edit : QTextEdit
            The QTextEdit displaying the text.
            
        """
        
        if path == "Unavailable":
            text = "File Unavailable"
        else:
            try:
                with open(path, "r") as text_file:
                    text = text_file.read()
            except (IOError, OSError, UnicodeDecodeError, 
                    SyntaxError) as e:
                self.gb.loading_error("text description", v2=True)
                text = "File Unavailable"
                    
        edit = qtw.QTextEdit(text)
        font = qtgui.QFont()
        font.setPointSize(10)
        edit.setFont(font)
        edit.setReadOnly(True)
        
        return edit
            
    class ScrollableWidget(qtw.QWidget):
        """A widget containing a scrollable matplotlib figure.     

        Parameters
        ----------
        index : int
            Tracks the number of open figures in the data set.
        window_geo : tuple of int
            The geometry (x, y, width, height) of the widget.
        icon : QIcon
            The window icon.
        fig_size : tuple of float
            The size (width, height) of the figure, in inches.
        min_size : tuple of int, optional
            The minimum size (width, height) of the canvas, in pixels.
            
        Attributes
        ----------
        fig : matplotlib.figure.Figure
            The main figure.
        canvas : FigureCanvas
            The canvas for `fig`.
        scroll : QScrollArea
            The scroll area containing `canvas`.
        nav : NavigationToolbar
            The toolbar for the figure.
        index : int
            Tracks the number of open figures in the data set.
        """
        

        def __init__(self, index, window_geo, icon, fig_size, 
                     min_size=None):
            
            qtw.QWidget.__init__(self)
            self.setWindowTitle("VEMOS")
            self.setWindowIcon(icon)
            
            self.fig = mpl.figure.Figure(figsize=(fig_size[0], fig_size[1]))
            #self.fig.set_tight_layout({"pad": 2})
            self.setLayout(qtw.QVBoxLayout())
            self.layout().setContentsMargins(0,0,0,0)
            self.layout().setSpacing(0)
            
            self.index=index

            self.canvas = FigureCanvas(self.fig)
            self.canvas.setParent(self)
            self.canvas.setFocusPolicy(qtcore.Qt.StrongFocus)
            self.canvas.setFocus()
            
            if min_size:
                self.canvas.setMinimumWidth(min_size[0])
                self.canvas.setMinimumHeight(min_size[1])
            self.canvas.setSizePolicy(qtw.QSizePolicy.MinimumExpanding,
                                     qtw.QSizePolicy.MinimumExpanding)
            
            self.scroll = qtw.QScrollArea(self)
            self.scroll.setWidget(self.canvas)
            self.scroll.setWidgetResizable(True)
    
            self.nav = NavigationToolbar(self.canvas, self)
            self.layout().addWidget(self.nav)
            self.layout().addWidget(self.scroll)
    
            self.setGeometry(*window_geo)
            self.show()
            
            
        def closeEvent(self,event):
            """Closes the current figure.
            
            Parameters
            ----------
            event : QEvent, optional
                The event that triggered closing the figure.
                
            Returns
            -------
            None
            
            """
            
            #plt.close()
            self.fig = None
            gc.collect()            