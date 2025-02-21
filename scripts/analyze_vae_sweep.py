import os
import numpy as np
import matplotlib.pyplot as plt
import json
import glob

def load_training_data(folder):
    """Load training and validation losses from a folder"""
    train_losses = np.loadtxt(os.path.join(folder, "training_losses.txt"))
    val_losses = np.loadtxt(os.path.join(folder, "validation_losses.txt"))
    return train_losses, val_losses

def parse_params_from_folder(folder):
    """Extract parameters from folder name"""
    name = os.path.basename(folder)
    beta = float(name.split('beta')[1].split('_')[0])
    warmup = int(name.split('warmup')[1].split('_')[0])
    loss_type = name.split('_')[-1]
    return beta, warmup, loss_type

def analyze_sweeps(base_folder):
    """Analyze all sweep results"""
    results = []
    
    # Find all sweep folders
    sweep_folders = glob.glob(os.path.join(base_folder, "vae_beta*"))
    
    for folder in sweep_folders:
        beta, warmup, loss_type = parse_params_from_folder(folder)
        train_losses, val_losses = load_training_data(folder)
        
        # Calculate metrics
        final_val_loss = val_losses[-1]
        min_val_loss = np.min(val_losses)
        convergence_epoch = np.argmin(val_losses)
        
        results.append({
            'beta': beta,
            'warmup': warmup,
            'loss_type': loss_type,
            'final_val_loss': final_val_loss,
            'min_val_loss': min_val_loss,
            'convergence_epoch': convergence_epoch
        })
        
        # Plot individual training curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title(f'β={beta}, warmup={warmup}, {loss_type}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(folder, 'training_curve.png'))
        plt.close()
    
    # Create summary plots
    plot_parameter_effects(results)
    
    # Save results
    with open(os.path.join(base_folder, 'sweep_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def plot_parameter_effects(results):
    """Create plots showing effects of different parameters"""
    # Convert results to numpy arrays for easier plotting
    betas = np.array([r['beta'] for r in results])
    warmups = np.array([r['warmup'] for r in results])
    min_losses = np.array([r['min_val_loss'] for r in results])
    
    # Plot beta vs loss for different warmups
    plt.figure(figsize=(10, 6))
    for w in np.unique(warmups):
        mask = warmups == w
        plt.plot(betas[mask], min_losses[mask], 'o-', label=f'warmup={w}')
    plt.xlabel('Beta')
    plt.ylabel('Min Validation Loss')
    plt.legend()
    plt.title('Effect of Beta and Warmup Steps on VAE Performance')
    plt.savefig('beta_warmup_effects.png')
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_folder', type=str, required=True,
                      help='Base folder containing sweep results')
    args = parser.parse_args()
    
    results = analyze_sweeps(args.sweep_folder)
    
    # Print best configuration
    best_result = min(results, key=lambda x: x['min_val_loss'])
    print("\nBest configuration:")
    print(f"Beta: {best_result['beta']}")
    print(f"Warmup steps: {best_result['warmup']}")
    print(f"Loss type: {best_result['loss_type']}")
    print(f"Minimum validation loss: {best_result['min_val_loss']:.4f}")
    print(f"Convergence epoch: {best_result['convergence_epoch']}") 