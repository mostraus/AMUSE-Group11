from amuse.lab import units, constants

import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from pathlib import Path




def EffectOnBoundPlot_fromData(InfoName="FullRun0ClusVel",PlotName="dVlPlot0ClusVel1"):

    directory = Path('DATA/')
    info_dict = {}
    bf_mins = []
    mass_ratios = []
    min_dists = []

    # Iterate over all items in the directory
    for file_path in directory.glob(f"{InfoName}*"):
        # Check if it is a file (and not a directory)
        if not file_path.is_file():
            print("ERROR on: ", file_path.name)
            continue
        print(f"Getting Info from {file_path.stem}")
        # get info from filename
        file_info = file_path.stem.split("_")
        a = file_info[3]
        b = float(file_info[4])
        m1 = float(file_info[-2]) | units.MSun
        m2 = float(file_info[-1][:-4]) | units.MSun

        if a not in info_dict:
            info_dict[a] = {"b":-1}

        if b >= info_dict[a]["b"]:
            if "PosAU" in file_path.stem:
                # here we get the position data
                pos_list = np.load(file_path)
                star_pos = pos_list[0,:,:]
                pert_pos = pos_list[1,:,:] - star_pos

                min_dist = np.min(np.linalg.norm(pert_pos, axis=1))
                info_dict[a]["min_dist"] = min_dist
                info_dict[a]["star_mass"] = m1
                info_dict[a]["perturber_mass"] = m2
                info_dict[a]["b"] = b
                info_dict[a]["star_pos"] = star_pos
                info_dict[a]["disk_pos"] = pos_list[2:,:,:]

            if "VelKMS" in file_path.stem:
                # here we get the velocity data
                vel_list = np.load(file_path)
                info_dict[a]["star_vel"] = vel_list[0,:,:]
                info_dict[a]["disk_vel"] = vel_list[2:,:,:]

    for key, inner_dict in info_dict.items():
        print(key)

        bf = []
        for i in range(1001):
            try:
                bf.append(get_bound_particles_fraction(inner_dict["star_mass"], inner_dict["star_pos"][i,:], inner_dict["star_vel"][i,:], inner_dict["disk_pos"][:,i,:], inner_dict["disk_vel"][:,i,:]))
            except Exception as e:
                print(e)

        #bf_mins.append(np.min(bf))
        bf_mins.append(bf[-1])
        mass_ratios.append(inner_dict["perturber_mass"] / inner_dict["star_mass"])
        min_dists.append(inner_dict["min_dist"])

    # turning the lists into np arrays
    bf_mins, mass_ratios, min_dists = np.array(bf_mins), np.array(mass_ratios), np.array(min_dists)
    
    # --- Disk Destruction vs Closest Separation Plot ---
    fig, ax = plt.subplots(figsize=(12,8))
    scat = ax.scatter(min_dists, bf_mins, c=mass_ratios, cmap="plasma")
    ax.hlines(0.5, min(min_dists), max(min_dists), linestyles="--")
    ax.set_xlabel("Closest Separation [AU]")
    ax.set_ylabel("Final Bound fraction")
    cbar = plt.colorbar(scat, ax=ax)
    cbar.set_label('Mass Ratio (M_p / M_s)')
    ax.set_title("Disk Destruction vs Closest Separation")
    fig.savefig(f"PLOT/{PlotName}.png")

    
    # --- Histogram of bound fraction of particles ---
    fig, ax = plt.subplots(figsize=(12,8))
    ax.hist(bf_mins)
    ax.set_xlabel("Fraction of bound particles")
    ax.set_ylabel("Amount")
    # --- calculate fraction of destroyed disks (bound frac < 0.5)
    # Convert dictionary values to a NumPy array first
    mask_destroyed = bf_mins < 0.5
    destroyed_frac = np.sum(mask_destroyed) / 100 #len(bf_mins)
    fig.suptitle(f"Fraction of destroyed disks (<0.5): {destroyed_frac}")
    fig.savefig(f"PLOT/Hist_{PlotName}.png")

    return bf_mins, mass_ratios, min_dists



def distVlostPlot_fromData(InfoName="FullRun0ClusVel",PlotName="dVlPlot0ClusVel1"):
    # Define the directory
    directory = Path('DATA/')
    info_dict = {}
    get_final_interaction_dict = {}
    prelim_min_bfs = {}

    # Iterate over all items in the directory
    for file_path in directory.glob(f"{InfoName}*"):
        # Check if it is a file (and not a directory)
        if not file_path.is_file():
            print(file_path.name)
            continue
        
        # get info from filename
        file_info = file_path.stem.split("_")
        a = file_info[3]
        b = file_info[4]
        m1 = file_info[-2] | units.MSun
        dict_string = f"{a}_{b}"

        if dict_string not in info_dict:
            info_dict[dict_string] = {}

        if "PosAU" in file_path.stem:
            # here we get the position data
            pos_list = np.load(file_path)
            pert_pos = pos_list[1,:,:]
            min_dist = np.min(np.linalg.norm(pert_pos, axis=1))
            info_dict[dict_string]["min_dist"] = min_dist
            info_dict[dict_string]["star_mass"] = m1
            info_dict[dict_string]["a"] = float(a)
            info_dict[dict_string]["b"] = float(b)
            info_dict[dict_string]["star_pos"] = pos_list[0,:,:]
            info_dict[dict_string]["disk_pos"] = pos_list[2:,:,:]

        if "VelKMS" in file_path.stem:
            # here we get the velocity data
            vel_list = np.load(file_path)
            info_dict[dict_string]["star_vel"] = vel_list[0,:,:]
            info_dict[dict_string]["disk_vel"] = vel_list[2:,:,:]

    fig, ax = plt.subplots(3,1,sharex=True, figsize=(12,8))
    for key, inner_dict in info_dict.items():
        print(key)
        #bf = [get_bound_particles_fraction(inner_dict["star_mass"], inner_dict["star_pos"][i,:], inner_dict["star_vel"][i,:], inner_dict["disk_pos"][:,i,:], inner_dict["disk_vel"][:,i,:]) for i in range(np.shape(inner_dict["star_pos"])[0])]
        bf = []
        for i in range(1001):
            try:
                bf.append(get_bound_particles_fraction(inner_dict["star_mass"], inner_dict["star_pos"][i,:], inner_dict["star_vel"][i,:], inner_dict["disk_pos"][:,i,:], inner_dict["disk_vel"][:,i,:]))
            except IndexError as e:
                print(e)

        if (inner_dict["a"] not in get_final_interaction_dict.keys()) or (inner_dict["b"] > get_final_interaction_dict[inner_dict["a"]]):
            prelim_min_bfs[inner_dict["a"]] = np.min(bf)
        bf_start = bf[0]
        bf_end = bf[-1]
        frac_lost = bf_end / bf_start
        if (inner_dict["min_dist"] > 7500) or (frac_lost > 2.5):
            print(f"Value too large: min_dist={inner_dict["min_dist"]}, frac_lost={frac_lost}")
            continue
        color = "blue"
        alpha = 1
        if inner_dict["b"] > 0:
            print("Second iteration")
            color = "green"
            alpha = 0.5
        ax[0].scatter(inner_dict["min_dist"], frac_lost, c=color, alpha=alpha)
        ax[1].scatter(inner_dict["min_dist"], bf_start, c=color, alpha=alpha)
        ax[2].scatter(inner_dict["min_dist"], bf_end, c=color, alpha=alpha)
    
    ax[0].set_ylabel("Fraction of Particles lost")
    ax[1].set_ylabel("Fraction of Particles bound at the beginning")
    ax[2].set_xlabel("Encounter Distance [AU]")
    ax[2].set_ylabel("Fraction of Particles bound at end")
    plt.tight_layout()
    fig.savefig(f"PLOT/{PlotName}.png")

    # --- Histogram of bound fraction of particles ---
    fig, ax = plt.subplots(figsize=(12,8))
    ax.hist(prelim_min_bfs.values())
    ax.set_xlabel("Fraction of bound particles")
    ax.set_ylabel("Amount")
    # --- calculate fraction of destroyed disks (bound frac < 0.5)
    # Convert dictionary values to a NumPy array first
    bfs_values = np.array(list(prelim_min_bfs.values()))
    mask_destroyed = bfs_values < 0.5
    destroyed_frac = np.sum(mask_destroyed) / len(prelim_min_bfs.values())
    fig.suptitle(f"Fraction of destroyed disks (<0.5): {destroyed_frac:.4f}")
    fig.savefig(f"PLOT/Hist_{PlotName}.png")
    #plt.show()
    
        


def get_bound_particles_fraction(star_mass, star_pos, star_vel, disk_positions, disk_velocities):
    """
    Calculates a boolean mask indicating which particles are gravitationally 
    bound to a central star.
    
    Arguments:
    star_mass -- Mass of the central star (AMUSE scalar quantity, e.g., 1 | units.MSun)
    relative_positions -- VectorQuantity of particle positions relative to the star (N, 3)
    relative_velocities -- VectorQuantity of particle velocities relative to the star (N, 3)
    
    Returns:
    is_bound -- Boolean numpy array where True indicates the particle is bound.
    """
    
    disk_positions -= star_pos
    disk_velocities -= star_vel
    # specific_kinetic_energy = 0.5 * v^2
    v_sq = np.linalg.norm(disk_velocities, axis=1)**2
    specific_kinetic = 0.5 * v_sq | units.kms**2
    # specific_potential_energy = -GM / r
    r = np.linalg.norm(disk_positions, axis=1) | units.AU
    specific_potential = - (constants.G * star_mass) / r
    
    # Calculate Total Specific Energy
    specific_total_energy = specific_kinetic + specific_potential

    
    # Check bound condition (Energy < 0)
    return sum(specific_total_energy < (0 | units.m**2 / units.s**2)) / len(disk_positions)


def append_row_to_csv(file_path, data_row, header_row=None):
    """
    Appends a list of data as a new row to an existing CSV file.
    
    Arguments:
    file_path -- String path to the .csv file
    data_row  -- List or iterable containing the data for the columns
    """
    # Check if file exists (optional, useful if you need to write headers first)
    file_exists = os.path.isfile(file_path)
    
    # Open in 'a' (append) mode
    # newline='' is required to prevent blank lines on Windows
    with open(file_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # Only write the header if the file didn't exist previously
        if not file_exists:
            writer.writerow(header_row)
        
        # Determine if we need to handle a specific delimiter? 
        # Default is comma.
        
        writer.writerow(data_row)


def write_bound_frac(M1, M2, pos_list, vel_list, ini_distance, times, disk_frac):
    bound_frac = np.zeros_like(times)
    for i,t in enumerate(times):
        bound_frac[i] = get_bound_particles_fraction(M1, pos_list[2:,i,:], vel_list[2:,i,:])    # 2: bc the 2 stars are the first 2 entries
    
    header_row = ["M_star1", "M_star2", "ini_distance", "disk_frac", "bound_frac_at_0Myr", "bound_frac_at_100Myr", "bound_frac_at_200Myr", "bound_frac_at_300Myr", "bound_frac_at_400Myr", "bound_frac_at_500Myr", "bound_frac_at_600Myr", "bound_frac_at_700Myr", "bound_frac_at_800Myr", "bound_frac_at_900Myr", "bound_frac_at_1000Myr"]
    data_row = [M1, M2, ini_distance, disk_frac, *bound_frac[::100]]
    while len(data_row) < len(header_row):
        data_row.append(-1.0)
    append_row_to_csv("bound_fraction.csv", data_row, header_row)


def calculate_energy(all_particles):
    kinetic = all_particles.kinetic_energy()
    potential = all_particles.potential_energy()
    return (kinetic + potential).value_in(units.J)


def plot_energy_evolution(times, energies, filename="EnergyEvolution.png"):
    fig, ax = plt.subplots(figsize=(12,8))
    mean = np.mean(energies)
    ax.scatter(times, energies/mean)
    #ax.hlines(mean, xmin=times[0], xmax=times[-1], colors="grey", linestyles="dashed", label="average")
    ax.set_xlabel("Time [yr]")
    ax.set_ylabel("Energy Deviation")
    ax.legend()
    ax.set_title("Energy Evolution of the System")

    fig.savefig(filename)


def bound_fraction_plot(enc_distances, frac_lost_list, PlotName="FracLostPlot"):
    fig, ax = plt.subplots()
    ax.scatter(enc_distances, frac_lost_list)
    ax.set_xlabel("Encounter Distance [AU]")
    ax.set_ylabel("Fraction of Particles lost")
    fig.savefig(f"PLOT/{PlotName}.png")


def combined_plot_DiskDestruction():
    plotname_tot = "CombinedRunTot"
    plotname1 = "CombinedRun0"
    fname1 = "FullRun0ClusVel"
    bf_mins1, mass_ratios1, min_dists1 = EffectOnBoundPlot_fromData(fname1, plotname1)
    plotname2 = "CombinedRun40"
    fname2 = "FullRunNew"
    bf_mins2, mass_ratios2, min_dists2 = EffectOnBoundPlot_fromData(fname2, plotname2)

    bf_mins = np.concatenate((bf_mins1, bf_mins2))
    mass_ratios = np.concatenate((mass_ratios1, mass_ratios2))
    min_dists = np.concatenate((min_dists1, min_dists2))

    # --- Disk Destruction vs Closest Separation Plot ---
    fig, ax = plt.subplots(figsize=(12,8))
    scat = ax.scatter(min_dists, bf_mins, c=mass_ratios, cmap="plasma")
    ax.hlines(0.5, min(min_dists), max(min_dists), linestyles="--")
    ax.set_xlabel("Closest Separation [AU]")
    ax.set_ylabel("Minimum Bound fraction")
    cbar = plt.colorbar(scat, ax=ax)
    cbar.set_label('Mass Ratio (M_p / M_s)')
    ax.set_title("Disk Destruction vs Closest Separation")
    fig.savefig(f"PLOT/{plotname_tot}.png")

    
    # --- Histogram of bound fraction of particles ---
    fig, ax = plt.subplots(figsize=(12,8))
    ax.hist(bf_mins)
    ax.set_xlabel("Fraction of bound particles")
    ax.set_ylabel("Amount")
    # --- calculate fraction of destroyed disks (bound frac < 0.5)
    # Convert dictionary values to a NumPy array first
    mask_destroyed = bf_mins < 0.5
    destroyed_frac = np.sum(mask_destroyed) / 200 # number of stars.in the combined cluster
    fig.suptitle(f"Fraction of destroyed disks (<0.5): {destroyed_frac:.4f}")
    fig.savefig(f"PLOT/Hist_{plotname_tot}.png")



if __name__ == "__main__":
    #distVlostPlot_fromData()
    combined_plot_DiskDestruction()