# ML Jets Summer2023

This is a workspace to implement ML-based jet studies.

## Basic setup
<details>
  <summary>Click for details</summary>
<br/> 
  
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

   
</details>

## Setup software environment for our analysis – on hiccup cluster
<details>
  <summary>Click for details</summary>
<br/> 
  
### Logon and allocate a node
  
Logon to hiccup:
```
ssh <user>@hic.lbl.gov
```
  
This brings you to the "login" node – the entry point from which we can request to be allocated a "compute" node to run our code. 

From the login node, we can request an interactive node from the slurm batch system:
   ```
   srun -N 1 -n 20 -t 24:00:00 -p std --pty bash
   ``` 
   which requests 1 full node (20 cores) for 2 hours in the `std` queue. You can choose the time and queue: you can use the `quick` partition for up to a 2 hour session, `std` for a 24 hour session, or `long` for a 72 hour session (but you may have to wait longer for the longer queues). 
Depending how busy the squeue is, you may get the node instantly, or you may have to wait awhile.
When you’re done with your session, just type `exit`.
Please do not run anything but the lightest tests on the login node. If you are finding that you have to wait a long time, let us know and we can take a node out of the slurm queue and logon to it directly.

### Initialize environment
  
Now we need to initialize the environment: load heppy (for Monte Carlo event generation and jet finding), set the python version, and create a virtual environment for python packages.
We have set up an initialization script to take care of this. 
The first time you set up, you can do:
```
cd ML_Jets_Summer2023
./init.sh --install
```
  
On subsequent times, you don't need to pass the `install` flag:
```
cd ML_Jets_Summer2023
./init.sh
```

Now we are ready to run our scripts.

   
</details>

## Run the data pipeline
  
We have constructed a simple pipeline to generate and analyze datasets. The initial version is minimal in terms of physics – it contains only basic illustrative functions for each step, and is meant to be experimented with and added to.

The data pipeline consists of the following steps:
1. Create dataset
   - Generate PYTHIA events (simulated proton-proton collisions)
   - Record relevant particle information from each event (e.g. jet reconstruction)
2. Load dataset and do ML analysis

The pipeline is steered by the script `steer_analysis.py`, where you can specify which parts of the pipeline you want to run, along with a config file `config.yaml`.

### To generate events and write the output to file:
```
python steer_analysis.py --generate --write
```

### To read events from file and do ML analysis:
```
python steer_analysis.py --read /path/to/training_data.h5 --analyze
```

### To generate events and do ML analysis ("on-the-fly"):
```
python steer_analysis.py --generate --analyze
```
