import torch
import numpy as np
from torch.utils.data import DataLoader
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QSpinBox, QGroupBox, QComboBox, QWidget
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches

class DatasetViewerDialog(QDialog):
    """Interactive Viewer for exploring the raw Training Dataset."""
    def __init__(self, dataset, alphabet, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.alphabet = alphabet
        self.setWindowTitle('EMNIST Dataset Viewer')
        self.setMinimumSize(600, 500)
        self.setStyleSheet("background-color: #1e1e2e; color: #EEEEEE;")

        self.class_indices = {i: [] for i in range(len(alphabet))}
        self.all_indices = list(range(len(dataset)))
        
        # Extract targets
        targets = dataset.targets
        if torch.is_tensor(targets):
            targets = targets.numpy()
            
        # Raw EMNIST targets are 1-26. Must subtract 1 to make them 0-25.
        for idx, label in enumerate(targets):
            adj_label = int(label) - 1
            if 0 <= adj_label < len(self.alphabet):
                self.class_indices[adj_label].append(idx)

        self.current_list = self.all_indices
        self._build_ui()
        self._update_view()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        controls = QGroupBox("Data Navigator")
        controls.setStyleSheet("QGroupBox { border: 1px solid #444; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; color: #4FC3F7; }")
        c_layout = QHBoxLayout(controls)

        self.filter_combo = QComboBox()
        self.filter_combo.addItem("All Classes")
        for letter in self.alphabet:
            self.filter_combo.addItem(f"Class: {letter}")
        self.filter_combo.currentIndexChanged.connect(self._on_filter_change)

        self.spin_box = QSpinBox()
        self.spin_box.setRange(0, len(self.current_list) - 1)
        self.spin_box.valueChanged.connect(self._update_view)

        btn_prev = QPushButton("◀ Prev")
        btn_next = QPushButton("Next ▶")
        btn_prev.clicked.connect(lambda: self.spin_box.setValue(self.spin_box.value() - 1))
        btn_next.clicked.connect(lambda: self.spin_box.setValue(self.spin_box.value() + 1))

        c_layout.addWidget(QLabel("Filter:"))
        c_layout.addWidget(self.filter_combo)
        c_layout.addWidget(QLabel(" Index:"))
        c_layout.addWidget(self.spin_box)
        c_layout.addWidget(btn_prev)
        c_layout.addWidget(btn_next)
        main_layout.addWidget(controls)

        self.fig_img = Figure(figsize=(4, 4), dpi=100, facecolor='#1e1e2e')
        self.canvas_img = FigureCanvas(self.fig_img)
        self.ax_img = self.fig_img.add_subplot(111)
        main_layout.addWidget(self.canvas_img)

        self.lbl_info = QLabel("")
        self.lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_info.setStyleSheet("font-size: 18px; font-weight: bold; color: #81C784;")
        main_layout.addWidget(self.lbl_info)

    def _on_filter_change(self, idx):
        if idx == 0:
            self.current_list = self.all_indices
        else:
            self.current_list = self.class_indices[idx - 1]
            
        self.spin_box.setMaximum(len(self.current_list) - 1)
        self.spin_box.setValue(0)
        self._update_view()

    def _update_view(self):
        if not self.current_list: return
        list_idx = self.spin_box.value()
        global_idx = self.current_list[list_idx]
        
        img, lbl = self.dataset[global_idx]
        
        self.ax_img.clear()
        self.ax_img.imshow(img.squeeze().numpy(), cmap='inferno')
        self.ax_img.axis('off')
        self.canvas_img.draw()
        
        self.lbl_info.setText(f"Label: {self.alphabet[lbl]} (Global Dataset Index: {global_idx})")


class PredictionViewerDialog(QDialog):
    """Interactive Window for exploring EMNIST model predictions with dynamic filters."""
    def __init__(self, model, dataset, device, alphabet, parent=None):
        super().__init__(parent)
        self.model = model
        self.dataset = dataset
        self.device = device
        self.alphabet = alphabet
        self.setWindowTitle('Interactive Prediction Explorer')
        self.setMinimumSize(900, 550)
        self.setStyleSheet("background-color: #1e1e2e; color: #EEEEEE;")

        self.correct_idxs = []
        self.incorrect_idxs = []
        self.all_idxs = list(range(len(dataset)))
        self._precompute_results()
        
        self.current_list = self.all_idxs
        self._build_ui()
        self._update_view()

    def _precompute_results(self):
        temp_loader = DataLoader(self.dataset, batch_size=512, shuffle=False)
        self.model.eval()
        idx_counter = 0
        with torch.no_grad():
            for X, y in temp_loader:
                X = X.to(self.device)
                if type(self.model).__name__ == "MultilayerPerceptron":
                    # Use .reshape instead of .view to handle permuted/non-contiguous tensors
                    X = X.reshape(X.size(0), -1)
                preds = self.model(X).argmax(dim=1).cpu()
                correct_mask = (preds == y)
                
                for is_correct in correct_mask:
                    if is_correct:
                        self.correct_idxs.append(idx_counter)
                    else:
                        self.incorrect_idxs.append(idx_counter)
                    idx_counter += 1

    def _build_ui(self):
        main_layout = QHBoxLayout(self)

        left_panel = QVBoxLayout()
        nav = QGroupBox("Navigator & Filter")
        nav.setStyleSheet("QGroupBox { border: 1px solid #444; } QGroupBox::title { color: #4FC3F7; }")
        nav_lay = QVBoxLayout(nav)

        self.filter_combo = QComboBox()
        self.filter_combo.addItems([
            f"All Predictions ({len(self.all_idxs)})", 
            f"🟢 Correct Only ({len(self.correct_idxs)})", 
            f"🔴 Incorrect Only ({len(self.incorrect_idxs)})"
        ])
        self.filter_combo.currentIndexChanged.connect(self._on_filter_change)

        spin_lay = QHBoxLayout()
        self.spin = QSpinBox()
        self.spin.setRange(0, len(self.current_list) - 1)
        self.spin.valueChanged.connect(self._update_view)
        
        btn_p = QPushButton("◀ Prev")
        btn_n = QPushButton("Next ▶")
        btn_p.clicked.connect(lambda: self.spin.setValue(self.spin.value() - 1))
        btn_n.clicked.connect(lambda: self.spin.setValue(self.spin.value() + 1))
        
        spin_lay.addWidget(QLabel("Index:"))
        spin_lay.addWidget(self.spin)
        
        btn_lay = QHBoxLayout()
        btn_lay.addWidget(btn_p)
        btn_lay.addWidget(btn_n)

        nav_lay.addWidget(self.filter_combo)
        nav_lay.addLayout(spin_lay)
        nav_lay.addLayout(btn_lay)
        left_panel.addWidget(nav)

        self.fig_img = Figure(figsize=(3, 3), dpi=100, facecolor='#1e1e2e')
        self.canvas_img = FigureCanvas(self.fig_img)
        self.ax_img = self.fig_img.add_subplot(111)
        self.ax_img.set_facecolor('#2a2a3e')
        left_panel.addWidget(self.canvas_img)

        self.lbl_pred = QLabel('Prediction: -')
        self.lbl_pred.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_panel.addWidget(self.lbl_pred)

        self.lbl_true = QLabel('True Label: -')
        self.lbl_true.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_true.setStyleSheet('font-size:14px; color: #EEEEEE;')
        left_panel.addWidget(self.lbl_true)

        main_layout.addLayout(left_panel, 1)

        self.fig_bar = Figure(figsize=(5, 6), dpi=100, facecolor='#1e1e2e')
        self.canvas_bar = FigureCanvas(self.fig_bar)
        self.ax_bar = self.fig_bar.add_subplot(111)
        self.ax_bar.set_facecolor('#2a2a3e')
        main_layout.addWidget(self.canvas_bar, 2)

    def _on_filter_change(self, idx):
        if idx == 0: self.current_list = self.all_idxs
        elif idx == 1: self.current_list = self.correct_idxs
        else: self.current_list = self.incorrect_idxs
            
        self.spin.setMaximum(max(0, len(self.current_list) - 1))
        self.spin.setValue(0)
        self._update_view()

    def _update_view(self):
        if not self.current_list:
            self.ax_img.clear(); self.ax_bar.clear(); self.canvas_img.draw(); self.canvas_bar.draw()
            return

        list_idx = self.spin.value()
        global_idx = self.current_list[list_idx]
        img, true_lbl = self.dataset[global_idx]

        self.model.eval()
        with torch.no_grad():
            X = img.unsqueeze(0).to(self.device)
            if type(self.model).__name__ == "MultilayerPerceptron":
                # Use .reshape instead of .view to handle permuted/non-contiguous tensors
                X = X.reshape(X.size(0), -1)
            out = self.model(X)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        
        pred = probs.argmax()
        ok = (pred == true_lbl)

        self.ax_img.clear()
        self.ax_img.imshow(img.squeeze().numpy(), cmap='gray')
        self.ax_img.axis('off')
        self.canvas_img.draw()

        color = '#81C784' if ok else '#EF9A9A'
        self.lbl_pred.setText(f'Pred: {self.alphabet[pred]} ({probs[pred]*100:.1f}%)')
        self.lbl_pred.setStyleSheet(f'font-size:22px; font-weight:bold; color:{color};')
        self.lbl_true.setText(f'True Label: {self.alphabet[true_lbl]}')

        self.ax_bar.clear()
        clrs = ['#81C784' if i == true_lbl else '#EF9A9A' if i == pred else '#4FC3F7' for i in range(26)]
        self.ax_bar.barh(self.alphabet, probs * 100, color=clrs)
        self.ax_bar.set_xlabel('Probability (%)', color='#EEEEEE')
        self.ax_bar.set_title('Class Probabilities', color='#EEEEEE', fontweight='bold')
        self.ax_bar.set_xlim(0, 110)
        self.ax_bar.invert_yaxis()
        self.ax_bar.tick_params(colors='#EEEEEE')
        
        for sp in self.ax_bar.spines.values(): sp.set_color('#444')
        self.fig_bar.tight_layout()
        self.canvas_bar.draw()