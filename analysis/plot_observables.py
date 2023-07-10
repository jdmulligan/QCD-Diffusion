import h5py
import matplotlib.pyplot as plt

# Load the generated data from the output file
output_file = "/software/users/diegohk/ML_Jets_Summer2023/analysis/TestOutput/training_data.h5"
with h5py.File(output_file, "r") as f:
    # Access the relevant datasets
    jet_pt = f["jet_pt"][:]
    jet_eta = f["jet_eta"][:]
    # Add more observables as needed

# Plot the observables
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# Plot jet pT
axs[0].hist(jet_pt, bins=50, range=(0, 100))
axs[0].set_xlabel("Jet pT")
axs[0].set_ylabel("Frequency")

# Plot jet eta
axs[1].hist(jet_eta, bins=50, range=(-5, 5))
axs[1].set_xlabel("Jet eta")
axs[1].set_ylabel("Frequency")

# Add more plots for other observables as needed

# Show the plots
plt.tight_layout()
plt.show()
