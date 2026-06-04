from PyQt6.QtWidgets import QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

# Import the actual, fully implemented pipeline tabs from their respective modules
from src.gui.tab_ann import ANNPipelineTab
from src.gui.tab_cnn import CNNPipelineTab

class InfoTab(QWidget):
    """
    A dedicated Information Tab to explain the application.
    """
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        
        info_text = """
        <h1 style='color: #4FC3F7;'>🧠 Multimodal Biomedical Data Analysis (MBDA)</h1>
        <h2>Final Project Dashboard</h2>
        <hr style='border: 1px solid #444;'>
        <p style='font-size: 16px; color: #EEEEEE;'>
        This interactive application explores two different Artificial Intelligence architectures for classifying the <b>EMNIST Letters</b> dataset (A-Z).
        </p>
        <ul style='font-size: 15px; color: #EEEEEE; line-height: 1.8;'>
            <li><b>Final Project 1 (ANN):</b> Utilizes a Multilayer Perceptron (MLP) with dynamic hidden layers, dropout regularization, and adaptive learning rate scheduling.</li>
            <li><b>Final Project 2 (CNN):</b> Employs a Convolutional Neural Network (EMNISTConvNet) designed to capture 2D spatial features, integrated with StepLR decay and dataset augmentation.</li>
        </ul>
        <p style='font-size: 14px; color: #888888;'>
        <b>Features Included:</b> Native CUDA acceleration, Background QThread training, interactive Matplotlib Pan/Zoom visualization, and real-time terminal logging.
        </p>
        """
        label = QLabel(info_text)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        label.setWordWrap(True)
        label.setStyleSheet("padding: 20px; background-color: #2a2a3e; border-radius: 8px;")
        
        layout.addWidget(label)
        layout.addStretch()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Window properties
        self.setWindowTitle("🧠 MBDA Final Projects Dashboard (ANN and CNN)")
        self.setGeometry(50, 50, 1300, 850) # Spacious size to fit Matplotlib dashboards
        
        # Main Tab Widget
        self.main_tabs = QTabWidget()
        self.setCentralWidget(self.main_tabs)
        
        # Instantiate Final Project 1 & 2 Tabs
        self.info_tab = InfoTab()
        self.ann_tab = ANNPipelineTab()
        self.cnn_tab = CNNPipelineTab()
        
        # Add to main layout
        self.main_tabs.addTab(self.info_tab, "ℹ️ Project Info")
        self.main_tabs.addTab(self.ann_tab, "🚀 Final Project 1: ANN (MLP)")
        self.main_tabs.addTab(self.cnn_tab, "🚀 Final Project 2: CNN")
        
        # Professional Dark Mode Stylesheet for the entire application
        self.setStyleSheet("""
            QMainWindow, QWidget { 
                background-color: #1e1e2e; 
                color: #EEEEEE; 
            }
            QTabWidget::pane { 
                border: 1px solid #444; 
            }
            QTabBar::tab {
                padding: 12px 25px; 
                font-weight: bold; 
                font-size: 14px;
                background-color: #2a2a3e; 
                border-top-left-radius: 4px; 
                border-top-right-radius: 4px;
                border: 1px solid #444; 
                border-bottom: none;
            }
            QTabBar::tab:selected { 
                background-color: #4FC3F7; 
                color: #1e1e2e; 
            }
            QPushButton {
                background-color: #4FC3F7; 
                color: #1e1e2e; 
                font-weight: bold; 
                padding: 10px;
                border-radius: 4px; 
                font-size: 13px;
            }
            QPushButton:hover { 
                background-color: #81C784; 
            }
            QPushButton:disabled { 
                background-color: #444444; 
                color: #888888; 
            }
            QLineEdit, QSpinBox {
                padding: 8px; 
                border: 1px solid #555; 
                border-radius: 4px; 
                background-color: #2a2a3e; 
                color: #FFF;
            }
            QProgressBar {
                border: 1px solid #444; 
                border-radius: 4px; 
                text-align: center; 
                font-weight: bold;
            }
            QProgressBar::chunk { 
                background-color: #81C784; 
            }
        """)