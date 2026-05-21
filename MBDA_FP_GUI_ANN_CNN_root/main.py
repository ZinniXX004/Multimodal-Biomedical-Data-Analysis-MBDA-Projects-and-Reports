import sys
from PyQt6.QtWidgets import QApplication
from src.gui.main_window import MainWindow

def main():
    # Initialize the Application
    app = QApplication(sys.argv)
    
    # Set a clean, cross-platform style
    app.setStyle("Fusion")
    
    # Initialize and show the Main Window
    window = MainWindow()
    window.show()
    
    # Execute the application loop safely
    sys.exit(app.exec())

if __name__ == "__main__":
    main()