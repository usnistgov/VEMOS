# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:12:38 2019

@author: Eve Fleisig

Opens the loading screen of VEMOS and, once data is loaded, opens the chosen 
interface.

If running a file directly is necessary, use this one. Otherwise, use 
__init__.py as per the installation instructions.
"""

import sys
import PyQt5.QtWidgets as qtw
import DataSet as DataSet
import VisualMetricAnalyzer as VisualMetricAnalyzer
import DataRecordBrowser as DataRecordBrowser

def run():
    app = qtw.QApplication(sys.argv)

    data_set = DataSet.DataSet()
    data_set.create_data_loading_widget()
    
    if data_set.interface_to_open == "Data Record Browser":
        drb = DataRecordBrowser.DataRecordBrowser(data_set)
    else:
        vma = VisualMetricAnalyzer.VisualMetricAnalyzer(data_set)        
    
    sys.exit(app.exec_())

run()