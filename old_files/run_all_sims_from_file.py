import pandas as pd
import numpy as np
from amuse.lab import *
from amuse.ext.protodisk import ProtoPlanetaryDisk
from HydroExample2 import simulate_hydro_disk, load_and_plot_data, load_and_animate_data, get_initial_values_new, simulate_disk_new, simulate_2disk_new
from functions import get_bound_particles_fraction, bound_fraction_plot


def run_all_cluster_sims(interactions_file):
    """
    Runs sequential hydrodynamic simulations of star-disk encounters based on 
    an interaction history file.

    This function iterates through a list of recorded stellar interactions. For each 
    unique star, it initializes a protoplanetary disk and sequentially simulates 
    the effect of passing perturbers (other stars) on that disk. It tracks the 
    evolution of the disk across multiple encounters, saves position/velocity data 
    for each run, and calculates mass loss metrics.

    Parameters
    ----------
    interactions_file : str
        Path to the CSV file containing the interaction history. 
        The file is expected to contain columns such as 'particle1_id' and 
        interaction parameters required by `get_initial_values_new`.

    Returns
    -------
    None
        This function does not return a value. It produces side effects including:
        - Saving AMUSE data files (.amuse) of the disks
        - Saving Numpy arrays (.npy) of position and velocity histories.
        - Generating diagnostic plots via `load_and_plot_data` and `bound_fraction_plot`.
    """
    # Load the interaction history
    df = pd.read_csv(interactions_file)

    # Identify the unique IDs of stars that host disks (the "target" stars)
    stars_with_interactions = np.unique(df["particle1_id"])

    # --- Configuration Constants ---
    # Parameters for the initialization of the protoplanetary disk
    N_disk = 2000               # Number of SPH particles in the disk
    M_disk=0.01 | units.MSun    # Mass of the disk
    R_min=1.0 | units.au        # Inner radius of the disk
    R_max=100.0 | units.au      # Outer radius of the disk
    q_out=-1.5                  # Power law index for surface density profile

    # Duration of the simulation for each specific encounter
    t_sim = 1000 | units.yr

    # --- State Tracking Variables ---
    # Used to detect and skip duplicate or extremely rapid sequential interaction logs
    last_interaction_time = 0 | units.yr
    last_star1_id = None
    last_star2_id = None

    # Storage for final analysis metrics
    enc_dists = []  # Minimum encounter distances
    frac_lost = []  # Fraction of disk particles unbound after encounter

    # --- Main Simulation Loop ---
    for idx, mc_star in enumerate(stars_with_interactions):
        # Create a subset of the dataframe for the current target star
        mask = (df["particle1_id"] == mc_star)
        interactions = df[mask]

        # Placeholders for the evolving state of the system between interactions
        temp_star = None
        temp_disk = None

        # Iterate through every recorded interaction for this specific star
        for i in range(len(interactions)):
            # Extract initial conditions for the encounter from the CSV row
            S1id, S2id, M1, M2, R1, R2, T, REL_DIST, REL_VEL = get_initial_values_new(interactions.iloc[i])
            print(f"Interaction between stars {idx} and {i} (IDs: {S1id}; {S2id})")

            # --- Skipping Logic ---
            # If the same pair interacts again within 1 Myr, treat it as a duplicate or 
            # ongoing event and skip to save computation time.
            if (last_star1_id == S1id) and (last_star2_id == S2id) and (T - last_interaction_time < 1|units.Myr):
                print(f"skip interaction {i} between {S1id} and {S2id} because only {T - last_interaction_time} have passed")
                continue

            # --- System Initialization ---
            if i == 0:
                # FIRST ENCOUNTER: Initialize a fresh Star and Disk
                star = Particles(1)
                star.mass = M1
                star.radius = R1 
                star.position = (0, 0, 0) | units.au    # Rest frame of this star
                star.velocity = (0, 0, 0) | units.kms
                star.name = "STAR" 
                
                hydro_converter = nbody_system.nbody_to_si(M1, R_max)

                disk = ProtoPlanetaryDisk(N_disk, 
                                convert_nbody=hydro_converter, 
                                Rmin=R_min/R_max, 
                                radius_min= R_min/R_max,
                                Rmax=1, 
                                radius_max= 1,
                                q_out=q_out, 
                                discfraction=M_disk/M1).result
            
            else:
                # SUBSEQUENT ENCOUNTERS: Use the evolved state from the previous run
                star = temp_star
                disk = temp_disk

            # --- Perturber Setup ---
            # Initialize the passing star (perturber) based on relative phase space coordinates
            perturber = Particles(1)
            perturber.mass = M2 
            perturber.radius = R2 
            perturber.position = REL_DIST   # Rest frame of other star
            perturber.velocity = REL_VEL 
            perturber.name = "PERTURBER"

            # Update tracking history
            last_interaction_time = T
            last_star1_id = S1id
            last_star2_id = S2id

            # --- Run Simulation ---
            # Define output filenames based on simulation parameters
            save_name = f"{T.value_in(units.yr)}Myr_{idx}_{i}_{t_sim.value_in(units.yr)}_{M1.value_in(units.MSun):.3f}_{M2.value_in(units.MSun):.3f}"
            
            # Execute the hydrodynamic simulation
            # Returns the evolved star and disk objects to be used in the next iteration (loop i+1)
            temp_star, temp_disk, pos_list, vel_list = simulate_disk_new(star, perturber, disk, f"DISK/DiskSave__{save_name}.amuse", t_sim=t_sim)
            
            # Save raw trajectory data
            filename_pos = f"DATA/FullRunNewPosAU__{save_name}.npy"
            filename_vel = f"DATA/FullRunNewVelKMS__{save_name}.npy"
            np.save(filename_pos, pos_list)
            np.save(filename_vel, vel_list)
            load_and_plot_data(filename_pos, filename_vel, PlotName=f"DEBUG_{idx}_{i}")

            # find closest approach and fraction of particles lost during this encounter
            min_dist = np.min(np.absolute(np.linalg.norm(pos_list[1], axis=1) - np.linalg.norm(pos_list[0], axis=1)))
            enc_dists.append(min_dist)
            bound_frac_end = get_bound_particles_fraction(M1, pos_list[0,-1,:], vel_list[0,-1,:], pos_list[2:,-1,:], vel_list[2:,-1,:])
            bound_frac_start = get_bound_particles_fraction(M1, pos_list[0,-1,:], vel_list[0,-1,:], pos_list[2:,0,:], vel_list[2:,0,:])
            # Avoid division by zero if bound_frac_start is 0
            if bound_frac_start > 0:
                frac_lost.append(bound_frac_end / bound_frac_start)
            else:
                frac_lost.append(0.0)

    bound_fraction_plot(enc_dists, frac_lost, PlotName=f"DistVSLost__{save_name}")


def run_sim_2disk(interactions_file, s1id, s2id, index=0, t_sim=500|units.yr):
    """
    Simulates a specific hydrodynamic interaction between two stars, each hosting 
    its own protoplanetary disk.

    This function extracts the initial kinematic conditions for a specific pair of 
    stars (identified by ID) from an interaction history file. It initializes SPH 
    models for both disks, transforms them into the correct relative frames, and 
    evolves the system over time.

    Parameters
    ----------
    interactions_file : str
        Path to the CSV file containing the interaction history (particle IDs, 
        masses, relative positions/velocities).
    s1id : int or str
        The ID of the primary star (particle 1).
    s2id : int or str
        The ID of the secondary star (particle 2).
    index : int, optional
        If these two stars interact multiple times in the history file, this index 
        selects which occurrence to simulate (default is 0, the first occurrence).
    t_sim : scalar(units.yr), optional
        The duration of the simulation (default is 500 years).

    Returns
    -------
    None
        Side effects include:
        - Saving `.npy` files containing position and velocity histories.
        - Generating a diagnostic plot via `load_and_plot_data`.
    """

    # --- Load and Filter Data ---
    df = pd.read_csv(interactions_file)

    # Filter for the specific pair of stars
    mask = ((df["particle1_id"] == s1id) & (df["particle2_id"] == s2id))
    df_filtered = df[mask]

    # Select the specific encounter instance (if they meet multiple times)
    arr = df_filtered.iloc[index]

    # get initial conditions for the simulation
    S1id, S2id, M1, M2, R1, R2, T, REL_DIST, REL_VEL = get_initial_values_new(arr)

    # --- Initialize Stars ---
    # Star 1: Placed at the origin (The "System Frame")
    star1 = Particles(1)
    star1.mass = M1
    star1.radius = R1 
    star1.position = (0, 0, 0) | units.au    # Rest frame of this star
    star1.velocity = (0, 0, 0) | units.kms
    star1.name = "STAR"

    # Star 2: Placed relative to Star 1
    star2 = Particles(1)
    star2.mass = M2 
    star2.radius = R2 
    star2.position = REL_DIST   # Rest frame of other star
    star2.velocity = REL_VEL 
    star2.name = "PERTURBER"

    # --- Configure Disk Parameters ---
    N_disk = 2000               # Number of SPH particles per disk 
    M_disk=0.01 | units.MSun    # Disk mass    
    R_min=1.0 | units.au        # Inner radius
    R_max=100.0 | units.au      # Outer radius
    q_out=-1.5                  # Surface density power law index

    hydro_converter1 = nbody_system.nbody_to_si(M1, R_max)
    hydro_converter2 = nbody_system.nbody_to_si(M2, R_max)

    # --- Generate Disks ---
    # Create Disk 1 (attached to Star 1, initialized at origin)
    disk1 = ProtoPlanetaryDisk(N_disk, 
                    convert_nbody=hydro_converter1, 
                    Rmin=R_min/R_max, 
                    radius_min= R_min/R_max,
                    Rmax=1, 
                    radius_max= 1,
                    q_out=q_out, 
                    discfraction=M_disk/M1).result
    
    # Create Disk 2 (initially created at origin relative to Star 2's mass)
    disk2 = ProtoPlanetaryDisk(N_disk, 
                    convert_nbody=hydro_converter2, 
                    Rmin=R_min/R_max, 
                    radius_min= R_min/R_max,
                    Rmax=1, 
                    radius_max= 1,
                    q_out=q_out, 
                    discfraction=M_disk/M1).result
    
    # Shift Disk 2 spatial coordinates to match Star 2's location/velocity
    # This ensures Disk 2 is orbiting Star 2, not Star 1
    disk2.position += REL_DIST
    disk2.velocity += REL_VEL

    # --- Run Simulation ---
    end_star, end_disk1, end_disk2, pos_list, vel_list = simulate_2disk_new(star1, star2, disk1, disk2, t_sim)

    # --- Save Output ---
    # Construct filename with physical parameters for easy identification
    save_name = f"{T.value_in(units.yr)}Myr_{S1id}_{S2id}_{t_sim.value_in(units.yr)}_{M1.value_in(units.MSun):.3f}_{M2.value_in(units.MSun):.3f}"
    filename_pos = f"DATA/2DiskRunPosAU__{save_name}.npy"
    filename_vel = f"DATA/2DiskRunVelKMS__{save_name}.npy"
    np.save(filename_pos, pos_list)
    np.save(filename_vel, vel_list)

    # Generate visualization
    load_and_plot_data(filename_pos, filename_vel, PlotName=f"2DiskRun_{S1id}_{S2id}")



def run_all_sims_1disk(interactions_file):
    df = pd.read_csv(interactions_file)

    main_character_stars = df["particle1_id"]
    partner_stars = df["particle2_id"]
    used_mc_stars = []
    t_sim = 200.0 | units.yr

    disk_filenames_dict = {}

    for i, mc in enumerate(main_character_stars):
        if i > 5:
            break
        p = partner_stars[i]
        print(f"RUNNING SIMULATION FOR {mc} AND {p}")
        if mc not in used_mc_stars:
            POSITIONS_LIST, VELOCITIES_LIST, info_array, disk_filename = simulate_hydro_disk(interactions_file, mc, p, t_sim=t_sim)
            used_mc_stars.append(mc)
            disk_filenames_dict[mc] = disk_filename
        else:   # to see how the disk of the other star is affected we can just do all simulations
        #elif mc in used_mc_stars and mc < p:
            index = np.sum(used_mc_stars == mc)
            POSITIONS_LIST, VELOCITIES_LIST, info_array, disk_filename = simulate_hydro_disk(interactions_file, mc, p, disk_file=disk_filenames_dict[mc], index=index, t_sim=t_sim)
            used_mc_stars.append(mc)
            disk_filenames_dict[mc] = disk_filename
        #else:   # mc in used stars and mc > p --> this combination has already been simulated
        #    print(f"This combination ({mc} and {p}) has already been simulated as {p} and {mc}")
        
        ### Save the data ###
        filename_pos = f"Data/DiskDataPosAU_{info_array[0]}Myr_{info_array[1]}_{info_array[2]}_{info_array[3]}yr_{info_array[4]}MSun_{info_array[5]}MSun.npy"
        filename_vel = f"Data/DiskDataVelKMS_{info_array[0]}Myr_{info_array[1]}_{info_array[2]}_{info_array[3]}yr_{info_array[4]}MSun_{info_array[5]}MSun.npy"
        np.save(filename_pos, POSITIONS_LIST)
        np.save(filename_vel, VELOCITIES_LIST)
        load_and_plot_data(filename_pos, filename_vel, PlotName=f"FullRun{i}")


def make_info_arr(df, S1_id, S2_id, index=0):
    mask = ((df['star_i'] == S1_id) & (df['star_j'] == S2_id))
    result = df[mask]
    arr = result.iloc[index]
    ### turn strings from csv into float values + units
    time = float(arr["time "][:-4]) | units.Myr  
    mass_1 = arr["mass_star_i_MSun"] | units.MSun
    mass_2 = arr["mass_star_j_MSun"] | units.MSun
    return time, mass_1, mass_2



if __name__ == "__main__":
    #run_all_sims_1disk("interactions.csv")
    #run_all_cluster_sims("Interactions stopping conditions_100Myr.csv")
    #run_sim_2disk("Interactions stopping conditions_100Myr.csv", 113871276108901526, 15793569915568245741, t_sim=500|units.yr)
    #fp1 = "Data//FullRunPosAU__99783.7073997Myr_1_0_500_9.331_2.330.npy"
    #fv1 = "Data//FullRunVelKMS__99783.7073997Myr_1_0_500_9.331_2.330.npy"
    #fp2 = "Data/FullRunPosAU__100266.667263Myr_1_1_500_9.331_2.330.npy"
    #fv2 = "Data/FullRunVelKMS__100266.667263Myr_1_1_500_9.331_2.330.npy"
    #fp3 = "Data/FullRunPosAU__100324.412464Myr_1_2_500_9.331_2.330.npy"
    #fv3 = "Data/FullRunVelKMS__100324.412464Myr_1_2_500_9.331_2.330.npy"
    #load_and_animate_data(fp1, fv1)
    #load_and_animate_data(fp2, fv2)
    #load_and_animate_data(fp3,fv3)
    fp = "Data/FullRunNewPosAU__14971755.7609Myr_7_1_1000_2.000_1.806.npy"
    fv = "Data/FullRunNewVelKMS__14971755.7609Myr_7_1_1000_2.000_1.806.npy"
    load_and_animate_data(fp, fv)


