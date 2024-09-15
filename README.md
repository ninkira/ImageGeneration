## DT8122 - Probabilistic Artificial Intelligence  
### Diffusion Model and Bayesian Flow Network Assignment  

This repository provides the code for the submission of DT8122 - Probabilistic Artificial Intelligence by Nina Christine Eckertz, Phd student @ NTNU.

## Repository Structure  

- *DT8122_eckertz_report.pdf* contains the project report.
- ZIP folders *diffusion/bfn_models* and *results_diffusion/bfn* contain the models and results obtained during the completion of the assignment.  
- Folders *models* and *results* are empty and intended for training new models and saving results. The code should automatically create the necessary subfolders. If not, please manually create subfolders called 'diffusion' or 'bfn'.  
- Scripts:
  - *visualisation.py* - Provides a class called `Visualisation` that contains the visualisation and plotting methods produced for this work.  
  - *diffusion_model.ipynb* - Contains the code for the implementation of the Diffusion Model.  
  - *bayesian_flow_network.ipynb* - Contains the code for the implementation of the Bayesian Flow Network.  

## Installation  

For both the Diffusion Model and the Bayesian Flow Network, this repository provides a Jupyter notebook for execution. Both Jupyter notebooks contain an installation block that installs the necessary libraries if needed. Hence, with a complete execution of the notebooks, the libraries will get installed automatically.  

**Note:** In case of execution in Google Colab, the folders *models* and *results* may need to be created manually, and the *visualisation.py* file needs to be uploaded manually into the local runtime.
