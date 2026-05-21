import time
import torch
import numpy as np
import os

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, device, output_dir):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        # Ensure output directory exists
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.output_dir, 'best_model.pth')
        
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'lr': []
        }
        self.best_val_loss = float('inf')

    def evaluate(self, loader):
        self.model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                
                # Forward pass (flatten X for MLP; later for CNN, this can be conditionally bypassed)
                logits = self.model(X) 
                loss = self.criterion(logits, y)
                
                preds = logits.argmax(dim=1)
                total_loss += loss.item() * X.size(0)
                total_correct += (preds == y).sum().item()
                total_samples += X.size(0)
                
        return total_loss / total_samples, total_correct / total_samples * 100

    def train(self, epochs, batch_callback=None, epoch_callback=None):
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss, running_correct, running_samples = 0.0, 0, 0
            epoch_start = time.time()

            for b_idx, (X_train, y_train) in enumerate(self.train_loader, 1):
                X_train = X_train.to(self.device, non_blocking=True)
                y_train = y_train.to(self.device, non_blocking=True)

                # Forward, Backward, Optimize
                self.optimizer.zero_grad()
                logits = self.model(X_train)
                loss = self.criterion(logits, y_train)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # Accumulate metrics
                preds = logits.argmax(dim=1)
                running_correct += (preds == y_train).sum().item()
                running_samples += X_train.size(0)
                running_loss += loss.item() * X_train.size(0)

                # Emit batch progress to GUI (if callback provided)
                if batch_callback and b_idx % 100 == 0:
                    batch_acc = running_correct / running_samples * 100
                    batch_loss = running_loss / running_samples
                    batch_callback(epoch, epochs, b_idx, len(self.train_loader), batch_loss, batch_acc)

            # Epoch Evaluation
            train_loss = running_loss / running_samples
            train_acc = running_correct / running_samples * 100
            val_loss, val_acc = self.evaluate(self.val_loader)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)

            # Save best model
            is_best = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.checkpoint_path)
                is_best = True

            epoch_time = time.time() - epoch_start
            
            # Emit epoch summary to GUI (if callback provided)
            if epoch_callback:
                epoch_callback(epoch, epochs, train_loss, train_acc, val_loss, val_acc, current_lr, epoch_time, is_best)

        total_time = time.time() - start_time
        return self.history, total_time