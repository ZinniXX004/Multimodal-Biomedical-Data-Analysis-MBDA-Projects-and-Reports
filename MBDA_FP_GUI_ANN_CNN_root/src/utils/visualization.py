import torch
import numpy as np
from matplotlib.figure import Figure

def plot_dataset_samples(dataset, alphabet, num_classes=26):
    """Displays 1 sample EMNIST grid from A-Z (as in Jupyter Notebook)."""
    fig = Figure(figsize=(14, 8), dpi=100)
    fig.patch.set_facecolor('#1e1e2e')
    fig.suptitle('Dataset Samples EMNIST Letters (A–Z)', fontsize=16, fontweight='bold', color='#EEEEEE', y=0.98)

    shown = {}
    for img, lbl in dataset:
        lbl = lbl.item() if isinstance(lbl, torch.Tensor) else lbl
        if lbl not in shown:
            shown[lbl] = img
        if len(shown) == num_classes: break

    for idx in range(num_classes):
        ax = fig.add_subplot(4, 7, idx + 1)
        ax.set_facecolor('#2a2a3e')
        if idx in shown:
            img = shown[idx].squeeze().numpy()
            ax.imshow(img, cmap='inferno')
            ax.set_title(alphabet[idx], fontsize=13, fontweight='bold', color='#EEEEEE')
        ax.axis('off')

    fig.tight_layout()
    return fig

def plot_prediction_inspection(model, test_loader, device, alphabet, n_show=20):
    """Displays grid of prediction results (Correct: Green, Incorrect: Red)."""
    model.eval()
    images_list, labels_list, preds_list, confs_list = [], [], [], []

    with torch.no_grad():
        for imgs, lbls in test_loader:
            X = imgs.to(device)
            # Handle ANN vs CNN input
            if type(model).__name__ == "MultilayerPerceptron":
                X = X.view(X.size(0), -1)
                
            outs = model(X)
            probs = torch.softmax(outs, dim=1)
            prd, conf_idx = torch.max(probs, 1)

            for i in range(len(lbls)):
                images_list.append(imgs[i].cpu().squeeze().numpy())
                labels_list.append(lbls[i].item())
                preds_list.append(conf_idx[i].item())
                confs_list.append(probs[i, conf_idx[i]].item() * 100)
                if len(images_list) >= n_show: break
            if len(images_list) >= n_show: break

    cols = 5
    rows = (n_show + cols - 1) // cols
    fig = Figure(figsize=(cols * 3, rows * 3.2), dpi=100)
    fig.patch.set_facecolor('#1e1e2e')
    fig.suptitle('Test Set Prediction Inspection', fontsize=15, fontweight='bold', color='#EEEEEE', y=0.98)

    for i in range(rows * cols):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_facecolor('#2a2a3e')
        
        if i >= len(images_list):
            ax.set_visible(False)
            continue

        img, true, pred, conf = images_list[i], labels_list[i], preds_list[i], confs_list[i]
        ok = (pred == true)

        ax.imshow(img, cmap='gray', interpolation='bilinear')
        color = '#81C784' if ok else '#EF9A9A'
        ax.set_title(f"True: {alphabet[true]}\nPred: {alphabet[pred]} ({conf:.0f}%)", 
                     fontsize=10, color=color, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(2.5)

    fig.tight_layout()
    return fig