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
import numpy as np
import os


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
        #     Displays some random samples from a data loader.
        fig, ax = plt.subplots(1, n, figsize=(3*n, 3))
        for i in range(n):
            ax[i].imshow(img[i][i], cmap='Greys_r', interpolation='nearest')
            ax[i].axis("off")
        title = title if title else f"{n} random samples"
        fig.suptitle(title, position=(0.5, 1.1))

    def visualize_noising_single_image_bfn(self, model, image, nb_steps=10, ema=any):
    
      # Prepare the grid with one less subplot, as we no longer plot the original image
      fig, axes = plt.subplots(1, nb_steps, figsize=(15, 3))

      print("Visualizing noising process for a single image without the original image.")

      # Initial condition for theta
      theta = torch.ones((1, model.D, model.D, model.K), device=image.device) / model.K

      # Iterate through noising steps
      for step in range(nb_steps):
          t = step / nb_steps
          t_tensor = t * torch.ones((theta.shape[0]), device=theta.device, dtype=theta.dtype)

          # Get the discrete output distribution
          k_probs = model.discrete_output_distribution(theta, t_tensor, ema=ema)

          # Sample k from the categorical distribution
          k = torch.distributions.Categorical(probs=k_probs).sample()

          # Calculate mean and std for noise
          alpha = model.beta * (2 * step + 1) / (nb_steps ** 2)  # Adjusted for step-based indexing
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
          axes[step].axis('off')

      plt.tight_layout()

    def visualize_noising_single_image_diffusion(self, pipeline, config, nb_steps):
      timesteps_to_visualize = [1, 5, 10, 25, 50, 75, 100, 200, 300, 400, 500, ]
      # Prepare the grid
      fig, axes = plt.subplots(1, len(timesteps_to_visualize), figsize=(15, 3))

      print("Visualizing noising process for a single image.")
      
      for idx, step in enumerate(timesteps_to_visualize):
          # Generate image at each timestep
          output = pipeline(
              batch_size=config.eval_batch_size,
              num_inference_steps=step,
              generator=torch.manual_seed(config.seed),
          )
          
          # Extract the image
          images = output.images
          image = images[0]  # Assumes the first image in the batch
          
          # Debugging: Check type and attributes of image

          if isinstance(image, torch.Tensor):
              print(f"Tensor shape: {image.shape}")
              image_array = image.permute(1, 2, 0).cpu().numpy()  # Convert tensor to NumPy array
          else:
              # Convert PIL image to NumPy array
              image_array = np.array(image)
          
          # Print image shape for debugging
          print(f"Image shape: {image_array.shape}")
          
          # Handle possible image shapes
          if image_array.ndim == 3 and image_array.shape[2] in [1, 3]:
              # Convert single channel grayscale image to (H, W)
              if image_array.shape[2] == 1:
                  image_array = image_array[:, :, 0]
          elif image_array.ndim == 3 and image_array.shape[2] == 4:
              # Convert RGBA to RGB
              image_array = image_array[:, :, :3]
          
          # Plot noisy image
          axes[idx].imshow(image_array, cmap='gray' if image_array.ndim == 2 else None)
          axes[idx].axis('off')
      
      plt.tight_layout()
      plt.show()


  

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

    def show_results(self, images, path):
      # Set up the figure and axes
      fig, ax = plt.subplots(4, 4, figsize=(10, 10))  # Adjust figsize as needed

      for i in range(16):
          # Convert the PIL image to a NumPy array
          image_array = np.array(images[i])
          
          # Display the image
          ax[i // 4, i % 4].imshow(image_array, cmap='Greys_r')
          ax[i // 4, i % 4].axis('off')
      plt.savefig(path, bbox_inches='tight')
      plt.show()


     

    def plot_loss(self, losses, path, title="Training Loss", xlabel="Epoch", ylabel="Loss"):

      plt.figure(figsize=(10, 6))
      plt.plot(losses, label="Loss")
      plt.xlabel(xlabel)
      plt.ylabel(ylabel)
      plt.title(title)
      plt.legend()
      plt.grid(True)
     
      plt.savefig(path,  bbox_inches='tight')
      plt.show()

    def plot_training_visualisation(self, sample_images, path):
      # increase the figure size for a larger plot
      plt.figure(figsize=(20, 5))
      
      for i, image_path in enumerate(sample_images):
  
          image = Image.open(image_path)
          
          # extract the filename from the path to use as the title
          filename = os.path.splitext(os.path.basename(image_path))[0]
          
          # plot the image in a subplot
          plt.subplot(1, len(sample_images), i + 1)
          plt.imshow(image)
          plt.title(f"{filename}")  # Use the filename as the title
          plt.axis("off")  # Hide the axis

      # show and safe
   
      plt.savefig(path)  
      plt.show()
