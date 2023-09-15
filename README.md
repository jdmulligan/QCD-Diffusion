# Conditional diffusion models for QCD confinement

This repository contains a proof-of-concept study using conditional diffusion models to learn the mapping of partons to hadrons in quantum chromodynamics (QCD).
Understanding this hadronization process is one of the biggest open questions in fundamental physics, since the confining interactions of QCD generate approximately 99% of the mass in the visible universe – and is key to the Clay Millenium Prize [Yang-Mills mass gap problem](https://www.claymath.org/millennium/yang-mills-the-maths-gap/).

<div align="center">
<img src="https://github.com/jdmulligan/QCD-Diffusion/assets/16219745/114be3cb-e2c1-476d-890d-e2128a2ae127" width="500" height="300">
</div>  
  
We implement:
- Generation of a simulated data set, consisting of paired images (partons, hadrons) where each pixel location represents an angular coordinate of a particle and the pixel intensity represents the energy of a particle.
- Training of conditional diffusion models to learn the forward or inverse hadronization mapping.

Using the [Shift-DDPM](https://arxiv.org/abs/2302.02373) implementation of conditional diffusion, we have been able to demonstrate a proof-of-concept that the ML model can successfully map different parton images to their corresponding hadron images. This result is highly nontrivial since the hadronization mapping is stochastic: the parton-to-hadron mapping is a one-to-many mapping. We show that, in our simplified setup, we can robustly invert any sampled hadron image into its single corresponding parton image, as shown below.
![image](https://github.com/jdmulligan/QCD-Diffusion/assets/16219745/7810cb92-69f7-4b0e-a7b0-e8a289656663)


## Running the analysis pipeline
  
The data pipeline consists of the following steps:
1. Create dataset
   - Generate PYTHIA events (simulated proton-proton collisions)
   - Record relevant particle information from each event (e.g. jet reconstruction)
2. Load dataset and do ML analysis

The pipeline is steered by the script `steer_analysis.py`, where you can specify which parts of the pipeline you want to run, along with a config file `config.yaml`.
 
Remember that you will first need to initialize the python virtual environment:
```
cd /path/to/QCD-Diffusion
source init.sh
```

### To generate a simulated data set and write it to file:
```
python analysis/steer_analysis.py --generate --write
```

### To read a simulated data set from file and do ML analysis:
```
python analysis/steer_analysis.py --read /path/to/training_data.h5 --analyze
```

### To generate a simulated data set and do ML analysis ("on-the-fly"):
```
python analysis/steer_analysis.py --generate --analyze
```

## Setup instructions for students
<details>
  <summary>Click for details</summary>
<br/> 

For reference, our reading and exercise list is linked here:
 - [Reading list](https://docs.google.com/document/d/1nDz0PvdvrQR79-z-nHU7dbMzTct1-O_NcjJzVuuaj5E/edit?usp=sharing)
 - [Google drive](https://drive.google.com/drive/u/0/folders/1eoGmWkVxYjx8As7fMrWZZGoWCt5Qil5U)
  
To begin, we need to set up a few things to be able to run our code and keep track of our changes with version control. Don't allow yourself to get stuck – if you are spending more than e.g. 10 minutes on a given step and are not sure what to do, ask one of us – don't hesitate.
  
We also encourage you to liberally use ChatGPT for software questions, both techincal (e.g. "How do I navigate to a certain directory on a linux terminal?", "I got this error after trying to do X: <paste error>") and conceptual ("Why do I want to use version control when writing code?", "What is a python virtual environment?"). 
  
To start, do the following:
  - Create a [GitHub](https://github.com) account
  - We will create an account for you on the `hiccup` cluster, a local computing cluster that we will use this summer. 
    - Open a terminal on your laptop and try to login: `ssh <user>@hic.lbl.gov`
      - Your home directory (`/home/<user>`) is where you can store your code
      - The `/rstorage` directory should be used to store data that you generate from your analysis (e.g. ML training datasets)
    - [generate an SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent?platform=linux) and upload it to your GitHub account
    - Clone this repository: `git clone <url>`
  - On your laptop, [download VSCode](https://code.visualstudio.com) 
    - Install the `Remote-SSH` extension – this will allow you to easily edit code on hiccup via your laptop's editor
    - Create a new workspace that ssh to hiccup, and add the folder for this repository to the workspace
    - Now, try to open a file and check that you can edit it successfully (with the changes being reflected on hiccup)
  
Now we are ready to set up the specific environment for our analysis.
  
### Setup software environment for our analysis – on hiccup cluster
  
If you are using the terminal inside of VSCode, you can logon to the hiccupgpu node by install the "Remote-SSH" extension in VSCode and adding a new remote server:
```
 Host hic.lbl.gov 
 ...
   Hostname hic.lbl.gov
   User <usr>
   Port 1142
```
 
Alternately, you can log directly onto the hiccup GPU node with:
```
ssh <user>@hic.lbl.gov -p 1142
```
  
Now we need to initialize the environment: load heppy (for Monte Carlo event generation and jet finding), set the python version, and create a virtual environment for python packages.
We have set up an initialization script to take care of this. 
The first time you set up, you can do:
```
cd ML_Jets_Summer2023
source init.sh --install
```
  
On subsequent times, you don't need to pass the `install` flag:
```
cd ML_Jets_Summer2023
source init.sh
```

Now we are ready to run our scripts.

   
</details>
