import matplotlib as plt
import torch
from datasets import load_dataset
import torchvision
import PIL
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from dataclasses import dataclass
import torch_ema
from torch.nn.functional import relu
from torch.optim import AdamW


#%matplotlib inline
class Visualisation:
    def __init__(self):
        plot_settings = {'ytick.labelsize': 16,
                        'xtick.labelsize': 16,
                        'font.size': 15,
                        'figure.figsize': (15, 3),
                        'axes.titlesize': 18,
                        'lines.linewidth': 1,
                        'lines.markersize': 2,
                        'legend.fontsize': 11,
                        'mathtext.fontset': 'stix',
                        'font.family': 'STIXGeneral'}
        plt.style.use(plot_settings)

    
    def show_samples(self, img, n: int, title: str=None) -> None:
        """
        Displays some random samples from a data loader.

        Args:
            dataloader (torch.utils.data.DataLoader): data loader storing the images.
            n (int): number of samples to display.
        """
        fig, ax = plt.subplots(1, n, figsize=(3*n, 3))
        for i in range(n):
            ax[i].imshow(img[i][i], cmap='Greys_r', interpolation='nearest')
            ax[i].axis("off")
        title = title if title else f"{n} random samples"
        fig.suptitle(title, position=(0.5, 1.1))

    def visualize_noising_single_image(self, model, image, nb_steps=10, ema=None):
    
        # Prepare the grid
        print("steps", nb_steps)
        fig, axes = plt.subplots(1, nb_steps + 1, figsize=(15, 3))

        print("Visualizing noising process for a single image.")

        # Plot the original image (Step 0)
        axes[0].imshow(image.permute(1, 2, 0).cpu().numpy(), cmap='gray')
        axes[0].set_title("Original")
        axes[0].axis('off')

        # Initial condition for theta
        theta = torch.ones((1, model.D, model.D, model.K), device=image.device) / model.K

        # Iterate through noising steps
        for step in range(1, nb_steps + 1):
            t = (step - 1) / nb_steps
            t_tensor = t * torch.ones((theta.shape[0]), device=theta.device, dtype=theta.dtype)

            # Get the discrete output distribution
            k_probs = model.discrete_output_distribution(theta, t_tensor, ema=ema)

            # Sample k from the categorical distribution
            k = torch.distributions.Categorical(probs=k_probs).sample()

            # Calculate mean and std for noise
            alpha = model.beta * (2 * step - 1) / (nb_steps ** 2)
            e_k = torch.nn.functional.one_hot(k, num_classes=model.K).float()
            mean = alpha * (model.K * e_k - 1)
            var = alpha * model.K
            std = torch.full_like(mean, fill_value=var).sqrt()

            # Generate noise
            eps = torch.randn_like(e_k)

            # Create noisy image
            y = mean + std * eps
            theta_prime = torch.exp(y) * theta
            theta = theta_prime / theta_prime.sum(-1, keepdim=True)

            # Convert theta to an image for visualization
            noisy_image = torch.argmax(theta, dim=-1).squeeze().cpu().numpy()

            # Plot noisy image
            axes[step].imshow(noisy_image, cmap='gray')
            axes[step].set_title(f"Step {step}")
            axes[step].axis('off')

        plt.tight_layout()
  

    def visualise_denoising_process(self, model, config, timesteps_to_visualize, ema):
                # create figure
        fig, axes = plt.subplots(1, len(timesteps_to_visualize), figsize=(15, 3)) 

        padding = 0.02
        grid_size = 3
        inset_size = (1 - padding * (grid_size + 1)) / grid_size

        for i, (ax, timestep) in enumerate(zip(axes,timesteps_to_visualize)): 
            images = model.sample(device=config.device.type, nb_steps=timestep, batch_size=9, ema=ema)
        
            for i in range(9):
                row = i // 3
                col = i % 3
                # Create an inset axis in the current subplot
                    # Compute the position for the inset axes with uniform padding
                x0 = padding + col * (inset_size + padding)
                y0 = 1 - (padding + (row -1 ) * (inset_size + padding))
                    
                # Create an inset axis in the current subplot
                inset_ax = ax.inset_axes([x0, y0, inset_size, inset_size])
                
                inset_ax.imshow(images[i], cmap='Greys_r')
                inset_ax.axis('off')
                ax.set_title(f"t: {timestep}")
                ax.axis("off")
        plt.tight_layout()
        plt.show()

    def show_results(self, images):
        plt.figure(figsize=(15,3))
        fig, ax = plt.subplots(3, 3, figsize=(2, 2),  layout='constrained')
        for i in range(9):
            ax[i // 3, i % 3].imshow(images[i], cmap='Greys_r')
            ax[i // 3, i % 3].axis('off')
        plt.show()