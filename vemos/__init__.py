"""
Opens the loading screen of VEMOS and, once data is loaded, opens the chosen
interface.
"""


import sys
import PyQt5.QtWidgets as qtw
import vemos.DataSet as DataSet
import vemos.VisualMetricAnalyzer as VisualMetricAnalyzer
import vemos.DataRecordBrowser as DataRecordBrowser

def run():
    """ Opens the loading screen and interfaces of VEMOS.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """

    app = qtw.QApplication(sys.argv)

    data_set = DataSet.DataSet()
    data_set.create_data_loading_widget()

    if data_set.interface_to_open == "Data Record Browser":
        drb = DataRecordBrowser.DataRecordBrowser(data_set)
    else:
        vma = VisualMetricAnalyzer.VisualMetricAnalyzer(data_set)

    sys.exit(app.exec_())
