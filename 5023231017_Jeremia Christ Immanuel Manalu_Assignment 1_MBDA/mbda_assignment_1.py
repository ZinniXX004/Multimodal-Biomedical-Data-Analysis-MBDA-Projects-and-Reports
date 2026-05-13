import sys
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib
# IMPORTANT: Set matplotlib backend to 'Agg' to avoid conflicts with PyQt6 GUI
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
# Added Navigation Toolbar for Interactive Plots (Zoom, Pan, Save, etc.)
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

# Import Scikit-Learn and XGBoost libraries
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score
from imblearn.over_sampling import SMOTE

# Import PyTorch libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import PyQt6 libraries
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QTextEdit, QTabWidget, QLabel, QScrollArea, QTableView)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QAbstractTableModel
from PyQt6.QtGui import QFont

# ==========================================
# 1. PANDAS TABLE MODEL FOR PYQT6
# ==========================================
class PandasModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            if isinstance(value, float):
                return f"{value:.4f}"
            return str(value)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._data.columns[section])
            if orientation == Qt.Orientation.Vertical:
                return str(self._data.index[section])
        return None

# ==========================================
# 2. PYTORCH ANN ARCHITECTURE
# ==========================================
class BreastCancerANN(nn.Module):
    def __init__(self):
        super(BreastCancerANN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.network(x)

# ==========================================
# 3. WORKER THREAD FOR MACHINE LEARNING PROCESS
# ==========================================
class MLWorker(QThread):
    log_signal = pyqtSignal(str)
    table_signal = pyqtSignal(str, object) 
    # Added int parameter to pass a custom minimum height for the canvas
    plot_signal = pyqtSignal(str, object, int)  
    finished_signal = pyqtSignal()

    def run(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_signal.emit(f"--- Using Device: {device} ---")

        # STEP 1: READ & CLEAN DATA
        self.log_signal.emit("\n--- STEP 1: Reading and Cleaning Data ---")
        try:
            data = pd.read_csv("BreastCancerData (4).csv")
        except Exception as e:
            self.log_signal.emit(f"ERROR reading file: {e}")
            self.finished_signal.emit()
            return
            
        data.drop(labels=['Unnamed: 32', 'id'], axis=1, inplace=True, errors='ignore')
        data['diagnosis'] = data['diagnosis'].replace({'B': 0, 'M': 1})
        
        self.log_signal.emit(f"Dataset Shape: {data.shape}")
        self.table_signal.emit("Raw Data Preview", data.head(100))

        # STEP 2: OVERALL EDA & HISTOGRAMS
        self.log_signal.emit("\n--- STEP 2: Overall Features EDA (Histograms, Heatmaps, Correlation) ---")
        all_features =[col for col in data.columns if col != 'diagnosis']
        
        # Histo Plot All
        n_cols = 5
        n_rows = int(np.ceil(len(all_features) / n_cols))
        fig_hist, axes = plt.subplots(n_rows, n_cols, figsize=(22, n_rows * 3.5))
        axes = axes.flatten()
        for idx, col in enumerate(all_features):
            sns.histplot(data=data, x=col, hue="diagnosis", stat="density", bins=30, kde=True,
                         palette=['#1f77b4', '#d62728'], element="step", ax=axes[idx])
            axes[idx].set_title(f'Dist: {col}', fontsize=10)
        for i in range(len(all_features), len(axes)):
            fig_hist.delaxes(axes[i])
        fig_hist.tight_layout()
        
        # Dynamically set height so it doesn't get squished (e.g., 6 rows * 250px = 1500px)
        dynamic_height = n_rows * 250 
        self.plot_signal.emit("1. Overall Features Distribution", fig_hist, dynamic_height)

        # Heatmap All
        fig_heat_all, ax_heat_all = plt.subplots(figsize=(20, 16))
        corr_matrix_all = data.corr()
        mask_all = np.zeros_like(corr_matrix_all, dtype=bool)
        mask_all[np.triu_indices_from(mask_all)] = True
        sns.heatmap(corr_matrix_all, vmin=-1, vmax=1, mask=mask_all, 
                    square=True, annot=True, fmt=".2f", annot_kws={"size": 7}, cmap="vlag", ax=ax_heat_all)
        ax_heat_all.set_title("Overall Features Heatmap")
        self.plot_signal.emit("2. Overall Features Heatmap", fig_heat_all, 800)

        # Sorted Correlation Table
        corr_with_diag = corr_matrix_all['diagnosis'].drop('diagnosis')
        abs_corr_with_diag = corr_with_diag.abs()
        sorted_features = abs_corr_with_diag.sort_values(ascending=False)
        df_sorted_corr = pd.DataFrame({
            'Feature': sorted_features.index,
            'Absolute Correlation': sorted_features.values,
            'Raw Correlation (Direction)': corr_with_diag[sorted_features.index].values
        })
        self.table_signal.emit("Feature Correlation Table", df_sorted_corr)

        # STEP 3: FEATURE SELECTION (TOP 8)
        self.log_signal.emit("\n--- STEP 3: Feature Selection (Top 8 Absolute) ---")
        features = list(sorted_features.head(8).index)
        self.log_signal.emit(f"Top 8 Selected Features: {features}")

        data_selected = data[features + ['diagnosis']]
        fig_heat_8, ax_heat_8 = plt.subplots(figsize=(8, 6))
        mask_selected = np.zeros_like(data_selected.corr(), dtype=bool)
        mask_selected[np.triu_indices_from(mask_selected)] = True
        sns.heatmap(data_selected.corr(), vmin=-1, vmax=1, mask=mask_selected, 
                    square=True, annot=True, cmap="vlag", ax=ax_heat_8)
        ax_heat_8.set_title("Top 8 Selected Features Heatmap")
        self.plot_signal.emit("3. Top 8 Features Heatmap", fig_heat_8, 500)

        # STEP 4 & 5: DATA SPLIT & K-FOLD CROSS VALIDATION
        self.log_signal.emit("\n--- STEP 4 & 5: Splitting Data & 5-Fold Cross Validation ---")
        X = data_selected[features].values
        y = data_selected['diagnosis'].values.astype(int)

        X_train_full, X_test_pure, y_train_full, y_test_pure = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        self.log_signal.emit(f"75% Training Data Size : {len(X_train_full)} rows")
        self.log_signal.emit(f"25% Pure Test Data Size : {len(X_test_pure)} rows")
        self.log_signal.emit("Running K-Fold (Please wait a moment...)")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_acc = {'RF':[], 'XGB': [], 'LR': [], 'NB':[], 'SVM': [], 'ANN':[]}
        
        for train_idx, val_idx in skf.split(X_train_full, y_train_full):
            X_tr, y_tr = X_train_full[train_idx], y_train_full[train_idx]
            X_v, y_v = X_train_full[val_idx], y_train_full[val_idx]
            
            X_tr_smote, y_tr_smote = SMOTE(random_state=42).fit_resample(X_tr, y_tr)
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr_smote)
            X_v_sc = scaler.transform(X_v)
            
            cv_acc['RF'].append(RandomForestClassifier(random_state=42).fit(X_tr_sc, y_tr_smote).score(X_v_sc, y_v))
            cv_acc['XGB'].append(XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss').fit(X_tr_sc, y_tr_smote).score(X_v_sc, y_v))
            cv_acc['LR'].append(LogisticRegression(random_state=42).fit(X_tr_sc, y_tr_smote).score(X_v_sc, y_v))
            cv_acc['NB'].append(GaussianNB().fit(X_tr_sc, y_tr_smote).score(X_v_sc, y_v))
            cv_acc['SVM'].append(SVC(kernel='linear', random_state=42).fit(X_tr_sc, y_tr_smote).score(X_v_sc, y_v))
            
            X_t = torch.FloatTensor(X_tr_sc).to(device)
            X_v_t = torch.FloatTensor(X_v_sc).to(device)
            y_t = torch.FloatTensor(y_tr_smote).unsqueeze(1).to(device)
            loader = DataLoader(TensorDataset(X_t, y_t), batch_size=32, shuffle=True)
            
            ann = BreastCancerANN().to(device)
            crit, opt = nn.BCEWithLogitsLoss(), optim.Adam(ann.parameters(), lr=0.001, weight_decay=1e-4)
            for _ in range(40):
                ann.train()
                for bx, by in loader:
                    opt.zero_grad()
                    loss = crit(ann(bx), by)
                    loss.backward()
                    opt.step()
            ann.eval()
            with torch.no_grad():
                cv_acc['ANN'].append(np.mean((torch.sigmoid(ann(X_v_t)).cpu().numpy().flatten() >= 0.5).astype(int) == y_v))

        self.log_signal.emit("\nAverage 5-Fold CV Accuracy:")
        for m_name, accs in cv_acc.items():
            self.log_signal.emit(f"- {m_name.ljust(15)} : {np.mean(accs)*100:.2f}%")

        # STEP 6: FINAL EVALUATION ON 25% PURE TEST DATA
        self.log_signal.emit("\n" + "="*50)
        self.log_signal.emit("STEP 6: FINAL EVALUATION ON 25% PURE TEST DATA")
        self.log_signal.emit("="*50)

        smote_final = SMOTE(random_state=42)
        X_train_full_smote, y_train_full_smote = smote_final.fit_resample(X_train_full, y_train_full)
        scaler_final = StandardScaler()
        X_train_final_scaled = scaler_final.fit_transform(X_train_full_smote)
        X_test_pure_scaled = scaler_final.transform(X_test_pure)

        # Evaluation Helper Function
        def eval_emit(y_true, y_pred, y_prob, model_name):
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            TN, FP, FN, TP = cm.ravel()
            acc = (TP + TN) / (TP + TN + FP + FN)
            sens = TP / (TP + FN) if (TP + FN) != 0 else 0
            spec = TN / (TN + FP) if (TN + FP) != 0 else 0
            prec = TP / (TP + FP) if (TP + FP) != 0 else 0
            f1 = f1_score(y_true, y_pred)
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            fig_eval, ax = plt.subplots(1, 2, figsize=(10, 4.5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Benign (0)', 'Malignant (1)'], yticklabels=['Benign (0)', 'Malignant (1)'], ax=ax[0])
            ax[0].set_title(f'Conf. Matrix: {model_name}')
            ax[0].set_ylabel('Actual')
            ax[0].set_xlabel('Predicted')
            
            ax[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
            ax[1].plot([0, 1],[0, 1], color='navy', lw=2, linestyle='--')
            ax[1].set_xlabel('False Positive Rate')
            ax[1].set_ylabel('True Positive Rate')
            ax[1].set_title(f'ROC Curve: {model_name}')
            ax[1].legend(loc="lower right")
            fig_eval.tight_layout()
            
            self.plot_signal.emit(f"Evaluation: {model_name}", fig_eval, 450)
            
            self.log_signal.emit(f"\n--- Metrics Output: {model_name} ---")
            self.log_signal.emit(f"TP: {TP} | FN: {FN} | FP: {FP} | TN: {TN}")
            self.log_signal.emit(f"Accuracy   : {acc*100:.2f}%")
            self.log_signal.emit(f"Sensitivity: {sens*100:.2f}% (Recall/TPR)")
            self.log_signal.emit(f"Specificity: {spec*100:.2f}% (TNR)")
            self.log_signal.emit(f"Precision  : {prec*100:.2f}%")
            self.log_signal.emit(f"F1-Score   : {f1*100:.2f}%")
            self.log_signal.emit(f"ROC AUC    : {roc_auc:.4f}")

        # Models Training & Eval
        models =[
            ("Random Forest", RandomForestClassifier(random_state=42)),
            ("XGBoost", XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')),
            ("Logistic Regression", LogisticRegression(random_state=42)),
            ("Naive Bayes", GaussianNB()),
            ("SVM", SVC(kernel='linear', probability=True, random_state=42))
        ]

        for name, clf in models:
            clf.fit(X_train_final_scaled, y_train_full_smote)
            pred = clf.predict(X_test_pure_scaled)
            prob = clf.predict_proba(X_test_pure_scaled)[:, 1]
            eval_emit(y_test_pure, pred, prob, name)

        # ANN FINAL & LEARNING CURVE
        self.log_signal.emit("\n--- Training PyTorch ANN ---")
        X_train_f_t = torch.FloatTensor(X_train_final_scaled).to(device)
        X_test_m_t  = torch.FloatTensor(X_test_pure_scaled).to(device)
        y_train_f_t = torch.FloatTensor(y_train_full_smote).unsqueeze(1).to(device)
        f_loader = DataLoader(TensorDataset(X_train_f_t, y_train_f_t), batch_size=32, shuffle=True)
        
        ann_f = BreastCancerANN().to(device)
        crit, opt = nn.BCEWithLogitsLoss(), optim.Adam(ann_f.parameters(), lr=0.001, weight_decay=1e-4)
        ep_losses, ep_accs = [],[]
        
        for epoch in range(100):
            ann_f.train()
            r_loss, cor, tot = 0.0, 0, 0
            for bx, by in f_loader:
                opt.zero_grad()
                out = ann_f(bx)
                loss = crit(out, by)
                loss.backward()
                opt.step()
                r_loss += loss.item() * bx.size(0)
                preds = (torch.sigmoid(out) >= 0.5).float()
                cor += (preds == by).sum().item()
                tot += by.size(0)
            ep_losses.append(r_loss / tot)
            ep_accs.append(cor / tot)

        fig_lc, ax_lc = plt.subplots(1, 2, figsize=(10, 3.5))
        ax_lc[0].plot(ep_losses, color='#d62728', lw=2)
        ax_lc[0].set_title('ANN: Learning Curve (Loss)')
        ax_lc[0].set_xlabel('Epoch')
        ax_lc[0].grid(True, alpha=0.3)
        ax_lc[1].plot(ep_accs, color='#2ca02c', lw=2)
        ax_lc[1].set_title('ANN: Learning Curve (Accuracy)')
        ax_lc[1].set_xlabel('Epoch')
        ax_lc[1].grid(True, alpha=0.3)
        fig_lc.tight_layout()
        self.plot_signal.emit("ANN Learning Curve", fig_lc, 400)

        ann_f.eval()
        with torch.no_grad():
            ann_prob = torch.sigmoid(ann_f(X_test_m_t)).cpu().numpy().flatten()
            ann_pred = (ann_prob >= 0.5).astype(int)
        eval_emit(y_test_pure, ann_pred, ann_prob, "PyTorch ANN")

        self.log_signal.emit("\n===== PROCESS COMPLETED =====")
        self.finished_signal.emit()

# ==========================================
# 4. MAIN GUI WINDOW (MODERN DARK THEME)
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Breast Cancer Diagnostic Analysis - Modern UI")
        self.setGeometry(100, 100, 1300, 850)
        self.apply_dark_theme()

        # Main Layout
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        
        # --- LEFT PANEL (CONTROLS & LOG) ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        title_label = QLabel("Breast Cancer Diagnostic Analysis")
        title_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #D4AF37; margin-bottom: 10px;") # Dark Yellow
        
        self.btn_run = QPushButton("▶ RUN ANALYSIS (TRAIN MODELS)")
        self.btn_run.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.btn_run.setMinimumHeight(40)
        self.btn_run.clicked.connect(self.start_analysis)
        
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        self.text_log.setFont(QFont("Courier New", 10))
        
        left_layout.addWidget(title_label)
        left_layout.addWidget(self.btn_run)
        left_layout.addWidget(QLabel("Progress Log / Terminal:"))
        left_layout.addWidget(self.text_log)
        
        # --- RIGHT PANEL (TABS) ---
        self.tabs = QTabWidget()
        
        # Tab 1: Data Tables
        self.tab_data = QWidget()
        self.layout_data = QVBoxLayout(self.tab_data)
        self.tabs.addTab(self.tab_data, "Data & Correlation Tables")
        
        # Tab 2: Visualizations (Plots)
        self.tab_plots = QWidget()
        self.layout_plots_scroll = QVBoxLayout(self.tab_plots)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        
        # This wrapper widget holds all the plots
        self.scroll_content = QWidget()
        self.layout_plots = QVBoxLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)
        self.layout_plots_scroll.addWidget(self.scroll_area)
        
        self.tabs.addTab(self.tab_plots, "Visualizations & Model Evaluation")
        
        # Combine
        main_layout.addWidget(left_panel, stretch=3)
        main_layout.addWidget(self.tabs, stretch=7)
        self.setCentralWidget(main_widget)

    def apply_dark_theme(self):
        # QSS Modern Dark with Dark Yellow (#D4AF37) and Blue (#007ACC) Highlights
        dark_qss = """
        QMainWindow { background-color: #1e1e1e; }
        QWidget { color: #f0f0f0; }
        QTextEdit { background-color: #2b2b2b; color: #00ff00; border: 1px solid #D4AF37; border-radius: 5px; padding: 5px;}
        QPushButton { background-color: #D4AF37; color: #1e1e1e; border: none; border-radius: 5px;}
        QPushButton:hover { background-color: #e5c158; }
        QPushButton:disabled { background-color: #555555; color: #888888; }
        QTabWidget::pane { border: 1px solid #007ACC; background-color: #1e1e1e; border-radius: 5px;}
        QTabBar::tab { background: #2b2b2b; padding: 10px 20px; margin-right: 2px; border-top-left-radius: 4px; border-top-right-radius: 4px;}
        QTabBar::tab:selected { background: #007ACC; color: white; font-weight: bold; }
        QTabBar::tab:hover:!selected { background: #3a3a3a; }
        QTableView { background-color: #2b2b2b; gridline-color: #555555; color: white; border: 1px solid #D4AF37;}
        QHeaderView::section { background-color: #D4AF37; color: #1e1e1e; font-weight: bold; padding: 4px; border: 1px solid #555555;}
        QScrollArea { border: none; }
        """
        self.setStyleSheet(dark_qss)

    def append_log(self, text):
        self.text_log.append(text)
        # Auto-scroll to bottom
        sb = self.text_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def add_table(self, title, df):
        lbl = QLabel(f"<b>{title}</b>")
        lbl.setStyleSheet("color: #007ACC; font-size: 14px; margin-top: 10px;")
        
        table_view = QTableView()
        model = PandasModel(df)
        table_view.setModel(model)
        table_view.resizeColumnsToContents()
        
        self.layout_data.addWidget(lbl)
        self.layout_data.addWidget(table_view)

    def add_plot(self, title, fig, min_height):
        # Create a container for the plot so label, toolbar, and canvas stick together
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        
        lbl = QLabel(f"<b>{title}</b>")
        lbl.setStyleSheet("color: #007ACC; font-size: 14px; margin-top: 15px;")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(min_height)
        
        # Add Matplotlib interactive toolbar (Zoom, Pan, Save, etc.)
        toolbar = NavigationToolbar(canvas, self)
        
        plot_layout.addWidget(lbl)
        plot_layout.addWidget(toolbar)
        plot_layout.addWidget(canvas)
        
        self.layout_plots.addWidget(plot_container)

    def start_analysis(self):
        self.btn_run.setEnabled(False)
        self.btn_run.setText("⏳ PROCESSING...")
        self.text_log.clear()
        
        # Clear previous layouts if re-running
        for i in reversed(range(self.layout_data.count())): 
            self.layout_data.itemAt(i).widget().setParent(None)
        for i in reversed(range(self.layout_plots.count())): 
            self.layout_plots.itemAt(i).widget().setParent(None)

        self.worker = MLWorker()
        self.worker.log_signal.connect(self.append_log)
        self.worker.table_signal.connect(self.add_table)
        self.worker.plot_signal.connect(self.add_plot)
        self.worker.finished_signal.connect(self.analysis_finished)
        self.worker.start()

    def analysis_finished(self):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶ RERUN ANALYSIS")

# ==========================================
# 5. ENTRY POINT APP
# ==========================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())