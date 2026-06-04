import matplotlib
matplotlib.use('qtagg') # Force PyQt backend

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QDialog
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=5, dpi=100, facecolor='#1e1e2e'):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor(facecolor)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)

class PlotWidget(QWidget):
    def __init__(self, parent=None, width=8, height=5):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.canvas = MplCanvas(self, width=width, height=height)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("background-color: #2a2a3e; border: 1px solid #444; border-radius: 4px;")
        
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)
    
    def update_plot(self):
        self.canvas.draw()

class PlotDialog(QDialog):
    def __init__(self, fig, title="Data Visualization", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1100, 800)
        self.setStyleSheet("background-color: #1e1e2e; color: #EEEEEE;")
        
        layout = QVBoxLayout(self)
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, self)
        toolbar.setStyleSheet("background-color: #2a2a3e; border: 1px solid #444; border-radius: 4px;")
        
        layout.addWidget(toolbar)
        layout.addWidget(canvas)