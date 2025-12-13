from amuse.lab import units, constants

import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from pathlib import Path



def distVlostPlot_fromData(InfoName="FullRunNew",PlotName="dVlPlot"):
    # Define the directory
    directory = Path('DATA/')
    info_dict = {}

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
            info_dict[dict_string]["b"] = float(b)
            info_dict[dict_string]["star_pos_start"] = pos_list[0,0,:]
            info_dict[dict_string]["star_pos_end"] = pos_list[0,-1,:]
            info_dict[dict_string]["disk_pos_start"] = pos_list[2:,0,:]
            info_dict[dict_string]["disk_pos_end"] = pos_list[2:,-1,:]

        if "VelKMS" in file_path.stem:
            # here we get the velocity data
            vel_list = np.load(file_path)
            info_dict[dict_string]["star_vel_start"] = vel_list[0,0,:]
            info_dict[dict_string]["star_vel_end"] = vel_list[0,-1,:]
            info_dict[dict_string]["disk_vel_start"] = vel_list[2:,0,:]
            info_dict[dict_string]["disk_vel_end"] = vel_list[2:,-1,:]

    fig, ax = plt.subplots(3,1,sharex=True, figsize=(12,8))
    for key, inner_dict in info_dict.items():
        print(key)
        bf_start = get_bound_particles_fraction(inner_dict["star_mass"], inner_dict["star_pos_start"], inner_dict["star_vel_start"], inner_dict["disk_pos_start"], inner_dict["disk_vel_start"])
        bf_end = get_bound_particles_fraction(inner_dict["star_mass"], inner_dict["star_pos_end"], inner_dict["star_vel_end"], inner_dict["disk_pos_end"], inner_dict["disk_vel_end"])
        frac_lost = bf_end / bf_start
        print(bf_start, bf_end, frac_lost)
        if (inner_dict["min_dist"] > 7500) or (frac_lost > 2.5):
            print(f"AAAAAAAAAAAAAH Value too large: min_dist={inner_dict["min_dist"]}, frac_lost={frac_lost}")
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


if __name__ == "__main__":
    distVlostPlot_fromData()
