"""
Opens the loading screen of VEMOS and, once data is loaded, opens the chosen
interface.

If running a file directly is necessary, use this one. Otherwise, use
__init__.py as per the installation instructions.
"""

import sys
import PyQt5.QtWidgets as qtw
from .DataSet import DataSet
from .VisualMetricAnalyzer import VisualMetricAnalyzer
from .DataRecordBrowser import DataRecordBrowser

def run():
    app = qtw.QApplication( sys.argv )

    data_set = DataSet()
    data_set.create_data_loading_widget()

    if data_set.interface_to_open == "Data Record Browser":
        drb = DataRecordBrowser( data_set )
    else:
        vma = VisualMetricAnalyzer( data_set )

    sys.exit(app.exec_())

run()
