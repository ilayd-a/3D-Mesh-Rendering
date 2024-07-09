# 3D-Mesh-Rendering

This is a basic program to get familiar with Pytorch3D. I used [this](https://www.thingiverse.com/thing:6665518) 3D model and converted it to an .obj file using an online converter. 


# How to run (for MacOS)?

1) Download the Anaconda installer for macOS from the [Anaconda website](https://repo.anaconda.com/archive/).
2) Open Terminal, switch to bash,  navigate to the download location, and run the installer:
   ```shell
     bash Anaconda3-2024.06-1-MacOSX-x86_64.sh
   ```
3) Initialize Anaconda:
   ```shell
     source ~/.bash_profile
   ```
4) Create a new conda environment:
     ```shell
        conda create -n pytorch3d_env python=3.9
      ```
5) Activate the environment:
   ```shell
      conda activate pytorch3d_env
   ```
6) Install PyTorch via Pip:
Make sure your Conda environment is activated and install PyTorch using pip.
   ```shell
      pip install torch torchvision torchaudio
   ```
7)Install PyTorch3D via Pip:
Install the required dependencies and PyTorch3D.
      
      ```shell
         pip install "git+https://github.com/facebookresearch/pytorch3d.git"
      ```
      
8) Then simply "python main.py"!!
