import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns

def run_evaluation_and_plot(model, test_loader, device, alphabet, history, figs):
    model.eval()
    all_preds, all_labels = [], []

    # 1. Run Inference
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device, non_blocking=True)
            
            # FIX 2: Use .reshape to prevent memory layout crashes
            if type(model).__name__ == "MultilayerPerceptron":
                X = X.reshape(X.size(0), -1)
                
            logits = model(X)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(y)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # 2. Calculate Metrics
    test_acc = accuracy_score(all_labels, all_preds) * 100
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    _, _, f1_per_class, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
    per_class_acc = [(all_preds[all_labels == i] == i).mean() * 100 for i in range(len(alphabet))]
    
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm) 

    # 3. Plotting Configurations
    DARK, PANEL, WHITE = '#1e1e2e', '#2a2a3e', '#EEEEEE'
    BLUE, ORG, GRN, RED = '#4FC3F7', '#FFB74D', '#81C784', '#EF9A9A'

    def style_ax(ax):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_color('#444')
        ax.tick_params(colors=WHITE, labelsize=9)
        ax.xaxis.label.set_color(WHITE)
        ax.yaxis.label.set_color(WHITE)
        ax.title.set_color(WHITE)
        ax.grid(True, alpha=0.15)
        return ax

    # PLOT 1: Training Curves
    fig_curves = figs['curves']
    fig_curves.clf()
    fig_curves.patch.set_facecolor(DARK)
    
    ax_loss = style_ax(fig_curves.add_subplot(1, 2, 1))
    ax_acc = style_ax(fig_curves.add_subplot(1, 2, 2))
    
    # Hide legends if no history exists (prevents UserWarning)
    if len(history['train_loss']) > 1:
        epochs_range = range(1, len(history['train_loss']) + 1)
        ax_loss.plot(epochs_range, history['train_loss'], 'o-', color=BLUE, lw=2, ms=4, label='Train Loss')
        ax_loss.plot(epochs_range, history['val_loss'], 's--', color=ORG, lw=2, ms=4, label='Val Loss')
        ax_loss.legend(labelcolor=WHITE, facecolor=PANEL)
        
        ax_acc.plot(epochs_range, [a/100 if a > 1 else a for a in history['train_acc']], 'o-', color=GRN, lw=2, ms=4, label='Train Acc')
        ax_acc.plot(epochs_range, [a/100 if a > 1 else a for a in history['val_acc']], 's--', color=RED, lw=2, ms=4, label='Val Acc')
        ax_acc.legend(labelcolor=WHITE, facecolor=PANEL)
    else:
        ax_loss.text(0.5, 0.5, "No Training History\n(Model Loaded from File)", color=WHITE, ha='center', va='center', fontsize=12)
        ax_acc.text(0.5, 0.5, "No Training History\n(Model Loaded from File)", color=WHITE, ha='center', va='center', fontsize=12)

    ax_loss.set_title('Loss Curve', fontweight='bold')
    ax_loss.set_xlabel('Epochs'); ax_loss.set_ylabel('Loss Value')

    ax_acc.set_title('Accuracy Curve', fontweight='bold')
    ax_acc.set_xlabel('Epochs'); ax_acc.set_ylabel('Accuracy Proportion')
    
    fig_curves.tight_layout()

    # PLOT 2: Per-Class Bar Chart
    fig_bars = figs['bars']
    fig_bars.clf()
    fig_bars.patch.set_facecolor(DARK)
    
    ax_bars = style_ax(fig_bars.add_subplot(111))
    colors_bar = [GRN if a >= 90 else ORG if a >= 75 else RED for a in per_class_acc]
    ax_bars.bar(alphabet, per_class_acc, color=colors_bar)
    ax_bars.axhline(test_acc, color='white', ls=':', lw=1.5, label=f'Avg {test_acc:.1f}%')
    ax_bars.set_title('Per-Class Accuracy (%)', fontweight='bold')
    ax_bars.set_xlabel('Alphabet Classes (A-Z)')
    ax_bars.set_ylabel('Accuracy (%)')
    ax_bars.set_ylim([0, 110])
    ax_bars.legend(labelcolor=WHITE, facecolor=PANEL)
    fig_bars.tight_layout()

    # PLOT 3: Confusion Matrix
    fig_cm = figs['cm']
    fig_cm.clf()
    fig_cm.patch.set_facecolor(DARK)
    
    ax_cm = fig_cm.add_subplot(111)
    ax_cm.set_facecolor(PANEL)
    
    heatmap = sns.heatmap(cm_norm, ax=ax_cm, cmap='YlOrRd', cbar=True, 
                          annot=True, fmt='.2f', annot_kws={"size": 7},
                          xticklabels=alphabet, yticklabels=alphabet)
    
    ax_cm.set_title('Confusion Matrix (Row-Normalized)', color=WHITE, fontweight='bold')
    ax_cm.set_xlabel('Predicted Label', color=WHITE)
    ax_cm.set_ylabel('True Label', color=WHITE)
    ax_cm.tick_params(colors=WHITE, labelsize=8)
    
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(colors=WHITE)
    
    fig_cm.tight_layout()

    # 4. Generate HTML Text Summary Box
    best_letter = alphabet[np.argmax(per_class_acc)]
    worst_letter = alphabet[np.argmin(per_class_acc)]
    
    summary_html = f"""
    <div style='background-color: #2a2a3e; padding: 15px; border-radius: 8px;'>
        <h2 style='color: #81C784; margin-top: 0;'>📊 Final Evaluation Summary</h2>
        <hr style='border: 1px solid #444;'>
        <table style='width: 100%; font-size: 15px; color: #EEEEEE;'>
            <tr>
                <td style='padding: 5px;'><b>Overall Accuracy:</b></td> <td style='padding: 5px;'>{test_acc:.3f}%</td> 
                <td style='padding: 5px;'><b>Best Letter:</b></td> <td style='padding: 5px; color:#81C784;'>{best_letter} ({max(per_class_acc):.1f}%)</td>
            </tr>
            <tr>
                <td style='padding: 5px;'><b>Macro Precision:</b></td> <td style='padding: 5px;'>{prec*100:.2f}%</td> 
                <td style='padding: 5px;'><b>Worst Letter:</b></td> <td style='padding: 5px; color:#EF9A9A;'>{worst_letter} ({min(per_class_acc):.1f}%)</td>
            </tr>
            <tr><td style='padding: 5px;'><b>Macro Recall:</b></td> <td style='padding: 5px;'>{rec*100:.2f}%</td> <td></td> <td></td></tr>
            <tr><td style='padding: 5px;'><b>Macro F1-Score:</b></td> <td style='padding: 5px;'>{f1*100:.2f}%</td> <td></td> <td></td></tr>
        </table>
    </div>
    """
    
    return summary_html, test_acc, prec, rec, f1