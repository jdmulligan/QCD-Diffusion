# ML Jets Summer2023

This is a workspace to implement ML-based jet studies.

## Setup software environment – on hiccup cluster
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
