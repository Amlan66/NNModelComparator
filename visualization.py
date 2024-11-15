import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import torch
import numpy as np

class TrainingVisualizer:
    def __init__(self):
        self.figure_size = (10, 5)
        self.test_figure_size = (2, 2)
        
    def create_loss_plot(self, losses):
        """Create loss plot and return as base64 image"""
        plt.figure(figsize=self.figure_size)
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Iterations (x100)')
        plt.ylabel('Loss')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf
    
    def create_test_results(self, model, device, test_loader, num_images=10):
        """Generate test results visualization for random test images"""
        model.eval()
        test_images = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                
                for i in range(min(num_images, len(data))):
                    img = data[i].cpu().numpy().squeeze()
                    pred_digit = pred[i].item()
                    true_digit = target[i].item()
                    
                    test_images.append(
                        self._create_single_test_image(img, pred_digit, true_digit)
                    )
                    
                    if len(test_images) == num_images:
                        break
                if len(test_images) == num_images:
                    break
        
        return ''.join(test_images)
    
    def _create_single_test_image(self, img, pred_digit, true_digit):
        """Create visualization for a single test image"""
        plt.figure(figsize=self.test_figure_size)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(f'Pred: {pred_digit}\nTrue: {true_digit}')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        return f'<div class="test-image"><img src="data:image/png;base64,{img_str}"></div>'

    def create_confusion_matrix(self, model, device, test_loader):
        """Create and return confusion matrix visualization"""
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.numpy())
        
        conf_matrix = np.zeros((10, 10), dtype=int)
        for p, t in zip(all_preds, all_targets):
            conf_matrix[t, p] += 1
        
        plt.figure(figsize=(8, 8))
        plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        plt.xticks(range(10))
        plt.yticks(range(10))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Add text annotations to the matrix
        for i in range(10):
            for j in range(10):
                plt.text(j, i, conf_matrix[i, j],
                        ha="center", va="center")
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        return buf