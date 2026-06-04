import os
import pandas as pd
import torch
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel, 
    QPushButton, QFormLayout, QLineEdit, QFileDialog, 
    QTextEdit, QProgressBar, QMessageBox, QScrollArea
)
from PyQt6.QtCore import Qt

from src.models.cnn import EMNISTConvNet
from src.engine.trainer import Trainer
from src.gui.plot_widget import PlotWidget, PlotDialog
from src.gui.tab_ann import TrainingWorker 
from src.datasets.emnist_loader import load_emnist_data
from src.engine.evaluator import run_evaluation_and_plot
from src.utils.visualization import plot_dataset_samples, plot_prediction_inspection
from src.gui.interactive_viewers import DatasetViewerDialog, PredictionViewerDialog

class CNNPipelineTab(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model, self.train_loader, self.test_loader = None, None, None
        
        self.output_dir = "outputs/cnn"
        self.ckpt_dir = os.path.join(self.output_dir, "checkpoints")
        self.logs_dir = os.path.join(self.output_dir, "logs")
        self.plots_dir = os.path.join(self.output_dir, "plots")
        for d in [self.ckpt_dir, self.logs_dir, self.plots_dir]:
            os.makedirs(d, exist_ok=True)
        
        self.header_label = QLabel(f"🖥️ Active Device: <b>{str(self.device).upper()}</b>")
        self.header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.header_label.setStyleSheet("font-size: 14px; padding: 10px; background-color: #2a2a3e; color: #81C784; border-radius: 5px;")
        self.layout.addWidget(self.header_label)

        self.cnn_tabs = QTabWidget()
        self.tab_config, self.tab_training, self.tab_eval = QWidget(), QWidget(), QWidget()
        self.cnn_tabs.addTab(self.tab_config, "⚙️ 1. Configuration & Data")
        self.cnn_tabs.addTab(self.tab_training, "🚀 2. Training Loop")
        self.cnn_tabs.addTab(self.tab_eval, "📊 3. Evaluation")
        self.layout.addWidget(self.cnn_tabs)
        
        self.setup_config_tab()
        self.setup_training_tab()
        self.setup_eval_tab()

    def setup_config_tab(self):
        layout = QFormLayout(self.tab_config)
        self.drop1_input, self.drop2_input = QLineEdit("0.5"), QLineEdit("0.3")
        self.epochs_input, self.lr_input = QLineEdit("15"), QLineEdit("0.001")
        self.step_size_input, self.gamma_input = QLineEdit("5"), QLineEdit("0.5")
        
        self.btn_load_data = QPushButton("📥 Load EMNIST Dataset (w/ Augmentation)")
        self.btn_view_samples = QPushButton("🖼️ View Dataset Samples")
        self.btn_build_model = QPushButton("🏗️ Build CNN Model")
        self.btn_load_model = QPushButton("📂 Load Saved Model (.pth)")
        
        self.btn_view_samples.setEnabled(False)
        self.btn_load_data.clicked.connect(self.load_dataset)
        self.btn_view_samples.clicked.connect(self.view_samples)
        self.btn_build_model.clicked.connect(self.build_model)
        self.btn_load_model.clicked.connect(self.load_saved_model)

        layout.addRow("Dropout 1 (FC1) / Dropout 2 (FC2):", QHBoxLayout())
        h_drops = QHBoxLayout()
        h_drops.addWidget(self.drop1_input); h_drops.addWidget(self.drop2_input)
        layout.addRow("Dropouts:", h_drops)
        layout.addRow("Epochs:", self.epochs_input)
        layout.addRow("Learning Rate (Adam):", self.lr_input)
        layout.addRow("StepLR Step Size / Gamma:", QHBoxLayout())
        h_lr = QHBoxLayout()
        h_lr.addWidget(self.step_size_input); h_lr.addWidget(self.gamma_input)
        layout.addRow("StepLR Params:", h_lr)
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_load_data)
        btn_layout.addWidget(self.btn_view_samples)
        btn_layout.addWidget(self.btn_build_model)
        btn_layout.addWidget(self.btn_load_model)
        layout.addRow("", btn_layout)
        
        self.status_label = QLabel("Status: Waiting for initialization...")
        layout.addRow(self.status_label)

    def load_dataset(self):
        self.btn_load_data.setEnabled(False)
        self.status_label.setText("Status: ⏳ Loading CNN Dataset...")
        try:
            self.train_loader, self.test_loader, self.alphabet = load_emnist_data(use_augmentation=True)
            self.status_label.setText(f"Status: ✅ Data loaded! {len(self.train_loader.dataset):,} train samples.")
            self.btn_view_samples.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
        finally:
            self.btn_load_data.setEnabled(True)

    def view_samples(self):
        if self.train_loader is None: return
        # Launch the interactive Dataset Viewer
        dialog = DatasetViewerDialog(self.train_loader.dataset, self.alphabet, parent=self)
        dialog.exec()

    def build_model(self):
        try:
            d1, d2 = float(self.drop1_input.text()), float(self.drop2_input.text())
            self.model = EMNISTConvNet(num_classes=26, drop1=d1, drop2=d2).to(self.device)
            self.status_label.setText(f"Status: ✅ CNN built! Params: {sum(p.numel() for p in self.model.parameters()):,}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def load_saved_model(self):
        if self.model is None:
            QMessageBox.warning(self, "Warning", "Please Build CNN Model first.")
            return
            
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model", self.ckpt_dir, "PyTorch Models (*.pth)")
        if file_path:
            try:
                checkpoint = torch.load(file_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                self.model.eval()
                self.status_label.setText(f"Status: 📂 Loaded weights from {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Error Loading Model", str(e))

    def setup_training_tab(self):
        layout = QVBoxLayout(self.tab_training)
        self.btn_start_train = QPushButton("▶️ Start Training CNN")
        self.btn_start_train.clicked.connect(self.start_training)
        
        self.train_progress = QProgressBar()
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet("background-color: #0c0c0c; color: #00ff00; font-family: Consolas, monospace; font-size: 13px; padding: 10px; border: 2px solid #333;")
        
        layout.addWidget(self.btn_start_train)
        layout.addWidget(self.train_progress)
        layout.addWidget(self.log_console)

    def start_training(self):
        if self.model is None or self.train_loader is None: return
        self.btn_start_train.setEnabled(False)
        self.log_console.clear()
        self.log_console.append("> root@mbda:~# ./train_cnn.sh\n> Initializing CUDA cores...\n> Starting CNN Training...")
        
        epochs, lr = int(self.epochs_input.text()), float(self.lr_input.text())
        step_sz, gamma_val = int(self.step_size_input.text()), float(self.gamma_input.text())
        self.train_progress.setMaximum(epochs * 100)
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_sz, gamma=gamma_val)
        
        trainer = Trainer(self.model, self.train_loader, self.test_loader, criterion, optimizer, scheduler, self.device, self.ckpt_dir)
        self.worker = TrainingWorker(trainer, epochs)
        self.worker.batch_update.connect(self.update_batch_log)
        self.worker.epoch_update.connect(self.update_epoch_log)
        self.worker.finished.connect(self.training_finished)
        self.worker.start()

    def update_batch_log(self, ep, eps, b, b_tot, loss, acc):
        self.train_progress.setValue(int(((ep - 1) / eps) * 100) + int((b / b_tot) * 100 / eps))
        if b % 100 == 0: self.log_console.append(f"> Epoch [{ep:02d}/{eps}] Batch [{b}/{b_tot}] Loss: {loss:.4f} Acc: {acc:.2f}%")

    def update_epoch_log(self, ep, eps, t_l, t_a, v_l, v_a, lr, t, is_best):
        self.log_console.append(f"\n[SYSTEM] Epoch {ep} OK ({t:.1f}s) | Val Acc: {v_a:.2f}% | LR: {lr:.6f}\n")

    def training_finished(self, history, total_time):
        self.btn_start_train.setEnabled(True)
        self.history = history
        self.train_progress.setValue(self.train_progress.maximum())
        self.log_console.append(f"> CNN Training completed in {total_time:.1f}s.")
        
        df_hist = pd.DataFrame(self.history)
        csv_path = os.path.join(self.logs_dir, 'training_history.csv')
        df_hist.to_csv(csv_path, index=False)
        self.log_console.append(f"> Training history successfully exported to: {csv_path}")

    def setup_eval_tab(self):
        layout = QVBoxLayout(self.tab_eval)
        btn_eval_layout = QHBoxLayout()
        self.btn_run_eval = QPushButton("🔍 Generate Interactive Dashboard")
        self.btn_inspect_preds = QPushButton("👀 Inspect Predictions (Samples)")
        self.btn_inspect_preds.setEnabled(False)
        
        self.btn_run_eval.clicked.connect(self.run_evaluation)
        self.btn_inspect_preds.clicked.connect(self.inspect_predictions)
        
        btn_eval_layout.addWidget(self.btn_run_eval)
        btn_eval_layout.addWidget(self.btn_inspect_preds)
        layout.addLayout(btn_eval_layout)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(scroll_content)
        
        self.summary_label = QLabel("Run evaluation to see results here.")
        self.summary_label.setStyleSheet("background-color: #2a2a3e; padding: 15px; border-radius: 5px;")
        
        # PREVENT SQUISHING: Set minimum heights!
        self.plot_curves = PlotWidget()
        self.plot_curves.setMinimumHeight(450)
        self.plot_bars = PlotWidget()
        self.plot_bars.setMinimumHeight(450)
        self.plot_cm = PlotWidget()
        self.plot_cm.setMinimumHeight(650)
        
        self.scroll_layout.addWidget(self.summary_label)
        self.scroll_layout.addWidget(self.plot_curves)
        self.scroll_layout.addWidget(self.plot_bars)
        self.scroll_layout.addWidget(self.plot_cm)
        
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

    def run_evaluation(self):
        if self.model is None or self.test_loader is None: return
        self.btn_run_eval.setEnabled(False)
        if not hasattr(self, 'history'):
            self.history = { 'train_loss': [0], 'val_loss': [0], 'train_acc': [0], 'val_acc': [0], 'lr': [0] }

        try:
            figs = {'curves': self.plot_curves.canvas.fig, 'bars': self.plot_bars.canvas.fig, 'cm': self.plot_cm.canvas.fig}
            summary_html, _, _, _, _ = run_evaluation_and_plot(self.model, self.test_loader, self.device, self.alphabet, self.history, figs)
            self.summary_label.setText(summary_html)
            self.plot_curves.update_plot()
            self.plot_bars.update_plot()
            self.plot_cm.update_plot()
            self.btn_inspect_preds.setEnabled(True)
            
            self.plot_curves.canvas.fig.savefig(os.path.join(self.plots_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
            self.plot_bars.canvas.fig.savefig(os.path.join(self.plots_dir, 'per_class_accuracy.png'), dpi=150, bbox_inches='tight')
            self.plot_cm.canvas.fig.savefig(os.path.join(self.plots_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
            
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
        finally:
            self.btn_run_eval.setEnabled(True)

    def inspect_predictions(self):
        if self.model is None or self.test_loader is None: return
        # Launch the interactive Prediction Explorer
        dialog = PredictionViewerDialog(self.model, self.test_loader.dataset, self.device, self.alphabet, parent=self)
        dialog.exec()