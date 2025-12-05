from amuse.lab import units, constants

import numpy as np
import csv
import os
import matplotlib.pyplot as plt

def get_bound_particles_fraction(star_mass, disk_positions, disk_velocities):
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

    ax.scatter(times, energies)
    ax.hlines(np.mean(energies), xmin=times[0], xmax=times[-1], colors="grey", linestyles="dashed", label="average")
    ax.set_xlabel("Time [yr]")
    ax.set_ylabel("Total Energy [J]")
    ax.legend()
    ax.set_title("Energy Evolution of the System")

    fig.savefig(filename)
