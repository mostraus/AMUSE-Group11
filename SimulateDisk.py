from amuse.lab import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from amuse.couple import bridge
from amuse.ext.protodisk import ProtoPlanetaryDisk
import pandas as pd

from functions import get_bound_particles_fraction, write_bound_frac, calculate_energy, plot_energy_evolution, bound_fraction_plot


# --- Simulation Codes ---
def simulate_disk_new(STAR, PERTURBER, DISK, disk_savename,
                      t_sim=500|units.yr, dt=1|units.yr):

    """
    Runs a coupled hydrodynamic and N-body simulation of a single protoplanetary disk 
    encountering a passing perturber star.

    This function sets up a "Bridge" between a gravity solver (Hermite) and a 
    hydrodynamics solver (Fi). It evolves the system, handles potential crashes 
    gracefully, records the trajectory history, and saves the final state to disk.

    Parameters
    ----------
    STAR : amuse.datamodel.particles.Particles
        A particle set containing the primary star (Host).
    PERTURBER : amuse.datamodel.particles.Particles
        A particle set containing the secondary star (Perturber).
    DISK : amuse.datamodel.particles.Particles
        The SPH particle set representing the gas disk around the primary star.
    disk_savename : str
        The filename (path) where the final state of the simulation (AMUSE format)
        will be saved. This allows the disk to be reused in subsequent simulations.
    t_sim : scalar(units.yr), optional
        The total duration of the simulation (default is 500 years).
    dt : scalar(units.yr), optional
        The time interval for data logging (default is 1 year).

    Returns
    -------
    tuple
        - star_primary (Particle): The updated particle object for the primary star.
        - disk (Particles): The updated set of particles for the disk.
        - POSITIONS_LIST (np.ndarray): 3D array (N, T, 3) of XYZ positions [AU].
        - VELOCITIES_LIST (np.ndarray): 3D array (N, T, 3) of Vxyz velocities [km/s].
    """
    # Converter for the hydro code (scaled to disk properties)
    hydro_converter = nbody_system.nbody_to_si(STAR.mass[0], 100|units.AU) #np.mean(DISK.position.value_in(units.AU))|units.AU)
        
    # Center everything on STAR
    PERTURBER.position += STAR.position
    PERTURBER.velocity += STAR.velocity

    # Particle set for all massive N-body objects
    stars = Particles()
    stars.add_particle(STAR)
    stars.add_particle(PERTURBER)

    # --- Setup Gravity Code (for stars) ---
    # Converter for the N-body code (scaled to the stellar system)
    gravity_converter = nbody_system.nbody_to_si(stars.mass.sum(), 400|units.AU) #np.linalg.norm(PERTURBER.position))
    
    gravity = Hermite(gravity_converter) 
    gravity.particles.add_particles(stars)
    ch2_stars = gravity.particles.new_channel_to(stars)

    # --- Setup Hydro Code (for gas disk) ---
    hydro = Fi(hydro_converter, mode="openmp")
    hydro.parameters.timestep = 0.05 | units.yr # Adjusted for disk timescale
    hydro.particles.add_particles(DISK)
    ch2_disk = hydro.particles.new_channel_to(DISK)

    # --- Setup Bridge ---
    gravhydro = bridge.Bridge(use_threading=False) # use_threading=False is often more stable
    gravhydro.add_system(gravity, (hydro,)) # Gravity (stars) acts on Hydro (disk)
    gravhydro.add_system(hydro, (gravity,)) # Hydro (disk) acts on Gravity (stars)
    
    gravhydro.timestep = 0.1*dt # Bridge timestep

    # --- Simulation Setup ---
    model_time = 0 | units.yr
    t_end = t_sim  # already has units from above
    
    times = np.linspace(0, t_end.value_in(units.yr), int(t_end/dt) + 1)

    N_disk = np.shape(DISK.mass)[0]

    POSITIONS_LIST = np.zeros((N_disk + 2, times.shape[0], 3))
    VELOCITIES_LIST = np.zeros((N_disk + 2, times.shape[0], 3))

    # --- For the energy calculation ---
    all_particles = ParticlesSuperset([stars, DISK])    # superset updates with its contents
    ENERGIES_J = np.zeros_like(times)

    # --- Simulation run ---
    for i,t in enumerate(times):
        model_time = t | units.yr
        try:
            gravhydro.evolve_model(model_time)
        except Exception as e:
            print(f"CRASH at time {model_time}")
            print(f"Number of gas particles: {np.shape(DISK.position)}")
            print(f"Number of star particles: {np.shape(stars.position)}")
            # If you want to see if any particle has huge velocity:
            print("Max velocity in Disk:", max(np.linalg.norm(DISK.velocity.in_(units.kms), axis=1)))
            print("Max distance in Disk:", max(np.linalg.norm(DISK.position.in_(units.AU), axis=1)))
            raise e # Re-raise the error so the script stops

        # Copy data back to particle sets
        ch2_disk.copy()
        ch2_stars.copy()

        ENERGIES_J[i] = calculate_energy(all_particles)
        
        print(f"t={model_time.in_(units.yr)}")
        POSITIONS_LIST[0][i][:] = np.array([stars[0].x.value_in(units.AU), stars[0].y.value_in(units.AU), stars[0].z.value_in(units.AU)])
        POSITIONS_LIST[1][i][:] = np.array([stars[1].x.value_in(units.AU), stars[1].y.value_in(units.AU), stars[1].z.value_in(units.AU)])
        VELOCITIES_LIST[0][i][:] = np.array([stars[0].vx.value_in(units.kms), stars[0].vy.value_in(units.kms), stars[0].vz.value_in(units.kms)])
        VELOCITIES_LIST[1][i][:] = np.array([stars[1].vx.value_in(units.kms), stars[1].vy.value_in(units.kms), stars[1].vz.value_in(units.kms)])
        for j in range(N_disk):
            p = DISK[j]
            POSITIONS_LIST[j+2][i][:] = np.array([p.x.value_in(units.AU), p.y.value_in(units.AU), p.z.value_in(units.AU)])
            VELOCITIES_LIST[j+2][i][:] = np.array([p.vx.value_in(units.kms), p.vy.value_in(units.kms), p.vz.value_in(units.kms)])

    # --- Save the disk for further runs with the same disk ---
    ch2_disk.copy()
    ch2_stars.copy()
    save_disk(all_particles, disk_savename)
    # --- Cleanup ---
    gravity.stop()
    hydro.stop()

    # --- Meta info extraction ---
    #plot_energy_evolution(times, ENERGIES_J, f"PLOT/EnergyEvol_{save_str}.png")
    #write_bound_frac(M1, M2, POSITIONS_LIST, VELOCITIES_LIST, REL_DIST, times, M_disk/M1)

    return stars[0], DISK, POSITIONS_LIST, VELOCITIES_LIST


def simulate_2disk_new(STAR, PERTURBER, DISK1, DISK2,
                      t_sim=500|units.yr, dt=1|units.yr):
   
    """
    Runs a coupled N-body and Hydrodynamic simulation of two stars, each with its own
    protoplanetary disk, interacting over a specified timeframe.

    This function sets up a "Bridge" system in AMUSE, coupling:
    1. A gravity solver (Hermite) for the stellar dynamics.
    2. A hydrodynamics solver (Fi) for the gas/disk particles.

    The stars affect the gas via gravity, and the gas affects the stars via gravity 
    (two-way coupling). Position and velocity data is logged at every timestep `dt`.

    Parameters
    ----------
    STAR : amuse.datamodel.particles.Particles
        A single-particle set representing the primary star (Host).
    PERTURBER : amuse.datamodel.particles.Particles
        A single-particle set representing the secondary star (Perturber).
    DISK1 : amuse.datamodel.particles.Particles
        The SPH particle set representing the disk around the primary star.
    DISK2 : amuse.datamodel.particles.Particles
        The SPH particle set representing the disk around the secondary star.
    t_sim : scalar(units.yr), optional
        The total physical time to evolve the simulation (default 500 yr).
    dt : scalar(units.yr), optional
        The interval at which data is logged to the output arrays (default 1 yr).

    Returns
    -------
    tuple
        A tuple containing the following elements in order:
        - star_primary (Particle): The updated particle object for the primary star.
        - disk1 (Particles): The updated set of particles for Disk 1.
        - disk2 (Particles): The updated set of particles for Disk 2.
        - POSITIONS_LIST (np.ndarray): A 3D array of shape (N_total, N_steps, 3) 
          containing XYZ positions in AU. Index 0 is Star1, Index 1 is Star2, 
          Indices 2+ are disk particles.
        - VELOCITIES_LIST (np.ndarray): A 3D array of shape (N_total, N_steps, 3) 
          containing Vxyz velocities in km/s.
    """
    # Converter for the hydro code (scaled to disk properties)
    hydro_converter = nbody_system.nbody_to_si(STAR.mass[0], 100|units.AU) #np.mean(DISK.position.value_in(units.AU))|units.AU)

    # Particle set for all massive N-body objects
    stars = Particles()
    stars.add_particle(STAR)
    stars.add_particle(PERTURBER)

    # --- Setup Gravity Code (for stars) ---
    # Converter for the N-body code (scaled to the stellar system)
    gravity_converter = nbody_system.nbody_to_si(stars.mass.sum(), 400|units.AU) #np.linalg.norm(PERTURBER.position))
    
    gravity = Hermite(gravity_converter) 
    gravity.particles.add_particles(stars)
    ch2_stars = gravity.particles.new_channel_to(stars)

    # --- Setup Hydro Code (for gas disk) ---
    hydro = Fi(hydro_converter, mode="openmp")
    hydro.parameters.timestep = 0.05 | units.yr # Adjusted for disk timescale
    hydro.particles.add_particles(DISK1)
    ch2_disk1 = hydro.particles.new_channel_to(DISK1)
    hydro.particles.add_particles(DISK2)
    ch2_disk2 = hydro.particles.new_channel_to(DISK2)

    # --- Setup Bridge ---
    gravhydro = bridge.Bridge(use_threading=False) # use_threading=False is often more stable
    gravhydro.add_system(gravity, (hydro,)) # Gravity (stars) acts on Hydro (disk)
    gravhydro.add_system(hydro, (gravity,)) # Hydro (disk) acts on Gravity (stars)
    
    gravhydro.timestep = 0.1*dt # Bridge timestep

    # --- Simulation Setup ---
    model_time = 0 | units.yr
    t_end = t_sim  # already has units from above
    
    times = np.linspace(0, t_end.value_in(units.yr), int(t_end/dt) + 1)

    N_disk1 = np.shape(DISK1.mass)[0]
    N_disk2 = np.shape(DISK2.mass)[0]

    POSITIONS_LIST = np.zeros((N_disk1 + N_disk2 + 2, times.shape[0], 3))
    VELOCITIES_LIST = np.zeros((N_disk1 + N_disk2 + 2, times.shape[0], 3))

    # --- For the energy calculation ---
    all_particles = ParticlesSuperset([stars, DISK1, DISK2])    # superset updates with its contents
    ENERGIES_J = np.zeros_like(times)

    # --- Simulation run ---
    for i,t in enumerate(times):
        model_time = t | units.yr

        gravhydro.evolve_model(model_time)

        # Copy data back to particle sets
        ch2_disk1.copy()
        ch2_disk2.copy()
        ch2_stars.copy()

        ENERGIES_J[i] = calculate_energy(all_particles)
        
        print(f"t={model_time.in_(units.yr)}")
        POSITIONS_LIST[0][i][:] = np.array([stars[0].x.value_in(units.AU), stars[0].y.value_in(units.AU), stars[0].z.value_in(units.AU)])
        POSITIONS_LIST[1][i][:] = np.array([stars[1].x.value_in(units.AU), stars[1].y.value_in(units.AU), stars[1].z.value_in(units.AU)])
        VELOCITIES_LIST[0][i][:] = np.array([stars[0].vx.value_in(units.kms), stars[0].vy.value_in(units.kms), stars[0].vz.value_in(units.kms)])
        VELOCITIES_LIST[1][i][:] = np.array([stars[1].vx.value_in(units.kms), stars[1].vy.value_in(units.kms), stars[1].vz.value_in(units.kms)])
        for j in range(N_disk1 + N_disk2):
            if j < N_disk1:
                p = DISK1[j]
            else:
                p = DISK2[j - N_disk1]

            POSITIONS_LIST[j+2][i][:] = np.array([p.x.value_in(units.AU), p.y.value_in(units.AU), p.z.value_in(units.AU)])
            VELOCITIES_LIST[j+2][i][:] = np.array([p.vx.value_in(units.kms), p.vy.value_in(units.kms), p.vz.value_in(units.kms)])

    # --- Cleanup ---
    gravity.stop()
    hydro.stop()

    # --- Meta info extraction ---
    #plot_energy_evolution(times, ENERGIES_J, f"PLOT/EnergyEvol_{save_str}.png")
    #write_bound_frac(M1, M2, POSITIONS_LIST, VELOCITIES_LIST, REL_DIST, times, M_disk/M1)

    return stars[0], DISK1, DISK2, POSITIONS_LIST, VELOCITIES_LIST


# --- Initializing Runs ---
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
            filename_pos = f"DATA/FullRun0ClusVelPosAU__{save_name}.npy"
            filename_vel = f"DATA/FullRun0ClusVelVelKMS__{save_name}.npy"
            np.save(filename_pos, pos_list)
            np.save(filename_vel, vel_list)
            load_and_plot_data(filename_pos, filename_vel, PlotName=f"FullRun0ClusVel_{idx}_{i}")

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

    bound_fraction_plot(enc_dists, frac_lost, PlotName=f"DistVSLost0ClusVel__{save_name}")


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


# --- Visualization ---
def load_and_animate_data(filename_pos, filename_vel):
    """
    Loads position data and creates an animation showing the evolution
    of all time steps.

    Args:
        filename_pos (str): Path to the NumPy file containing position data.
        filename_vel (str): Path to the NumPy file containing velocity data.
    """
    
    # --- 1. Load Data ---
    POSITIONS_LIST = np.load(filename_pos)
    VELOCITIES_LIST = np.load(filename_vel)
    
    # POSITIONS_LIST shape is assumed to be (N_particles, N_timesteps, 3)
    N_timesteps = POSITIONS_LIST.shape[1]
    
    # Separate data components
    star1_positions = POSITIONS_LIST[0, :, :]
    star1_velocities = VELOCITIES_LIST[0, :, :]
    star2_positions = POSITIONS_LIST[1, :, :]
    disk_positions = POSITIONS_LIST[2:, :, :]
    disk_velocities = VELOCITIES_LIST[2:, :, :]

    # Pre-Calculate all speeds
    disk_speeds_all = np.linalg.norm(disk_velocities, axis=2)
    v_min = disk_speeds_all.min()
    v_max = disk_speeds_all.max()
    
    # --- 2. Setup Figure and Initial Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    lim = 300 # Set plot limits
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    s = filename_pos.split("_")   
    ax.set_title(f"Interaction between Masses {s[-2]} and {s[-1][:-4]} at Cluster Time: {s[2][:-3]}yr")
    
    # Initialize the plot elements
    # Star 1 (Red, centered at 0,0 after subtraction)
    star1_plot = ax.scatter(0, 0, c='blue', s=100) 
    
    # Star 2 (Blue, relative to Star 1)
    star2_plot = ax.scatter(0, 0, c='green', s=100) 
    
    # Disk particles (Black/Gray)
    disk_plot = ax.scatter([], [], c=[], cmap='hot', vmin=v_min, vmax=v_max, s=1, alpha=0.3) 

    # Add colorbar
    cbar = plt.colorbar(disk_plot, ax=ax)
    cbar.set_label('Velocity [km/s]')
    
    # Text element for time step
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    #bound_frac_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='bottom')
    #z_dist_text = ax.text(1, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='bottom')


    # --- 3. Animation Update Function ---
    def update_frame(i):
        """
        Updates the plot elements for frame i.
        """
        # Get the position of Star 1 at the current time step i
        x0, y0, z0 = star1_positions[i]
        
        # --- Update Disk Particles ---
        # Calculate disk positions relative to Star 1
        disk_x_rel = disk_positions[:, i, 0] - x0
        disk_y_rel = disk_positions[:, i, 1] - y0
        
        # Update the data in the existing scatter plot object
        disk_plot.set_offsets(np.column_stack([disk_x_rel, disk_y_rel]))
        
        # --- Update Star 2 ---
        # Calculate Star 2 position relative to Star 1
        star2_x_rel = star2_positions[i, 0] - x0
        star2_y_rel = star2_positions[i, 1] - y0
        star2_z_rel = star2_positions[i, 2] - z0
        
        # Star 1 is always at (0, 0) in the relative frame
        star2_plot.set_offsets(np.column_stack([star2_x_rel, star2_y_rel]))
        star1_plot.set_offsets(np.column_stack([0, 0]))

        # Update colors
        current_speeds = disk_speeds_all[:, i]
        disk_plot.set_array(current_speeds)
        # Update the time step text
        bf = get_bound_particles_fraction(float(s[6]) | units.MSun, star1_positions[i,:], star1_velocities[i,:], disk_positions[:,i,:], disk_velocities[:,i,:])

        time_text.set_text(f"Time Step: {i} / {N_timesteps - 1} \nBound Fraction: {bf:.3f} \nperturber z-dist: {star2_z_rel:.1f}au")
        #bound_frac_text.set_text(f"Bound Fraction: {bf:.3f}")
        # Update z_dist_text
        #z_dist_text.set_text(f"perturber z-dist: {star2_z_rel:.1f}au")
        
        # Return the elements that have changed
        return star1_plot, star2_plot, disk_plot, time_text

    # --- 4. Create and Save Animation ---
    
    # Note: interval is the delay between frames in milliseconds.
    # The frames argument is the number of steps to animate (all steps here).
    anim = animation.FuncAnimation(
        fig, 
        update_frame, 
        frames=N_timesteps, 
        interval=50, # 50 ms per frame = 20 frames per second (fps)
        blit=True     # Optimized drawing
    )
    output_filename = f"ANIM/DiskAnimation{filename_pos[18:-4]}.mp4"
    print(f"Saving animation to {output_filename}...")
    
    # Use 'ffmpeg' writer for MP4. You might need to install it on your system.
    # Use 'imagemagick' writer for GIF (slower).
    
    anim.save(output_filename, writer='ffmpeg', fps=20) 

    print("Animation saved!")
    plt.close(fig) # Close the figure to free up memory
    
    return anim # Optionally return the animation object


def load_and_plot_data(filename_pos, filename_vel, PlotName="DiskPlot"):
    """
    Loads simulation trajectory data and generates a 4x4 grid of snapshots 
    visualizing the time evolution of the system.

    This function reads NumPy arrays containing position and velocity histories,
    calculates a stride to capture 16 evenly spaced snapshots, and plots the 
    system state centered on the primary star. The disk particles are colored 
    according to their velocity magnitude.

    Parameters
    ----------
    filename_pos : str
        Path to the .npy file containing the position history. 
        Shape expected: (N_particles, N_timesteps, 3).
        Index 0 is assumed to be the Primary Star; Index 1 the Secondary Star.
    filename_vel : str
        Path to the .npy file containing the velocity history.
        Shape expected: (N_particles, N_timesteps, 3).
    PlotName : str, optional
        Prefix for the output image filename (default is "DiskPlot").

    Returns
    -------
    None
        Saves a PNG image to the 'PLOT/' directory.

    Notes
    -----
    - **Coordinate System:** The plots are transformed to the rest frame of 
      Star 1 (the primary). Star 1 is always at (0,0).
    - **Filename Parsing:** The function assumes `filename_pos` contains a double 
      underscore `__` to separate metadata, used for generating the output filename.
    - **Axis Limits:** The plots are hardcoded to a fixed view of +/- 350 AU.
    """

    # --- Load Data ---
    POSITIONS_LIST = np.load(filename_pos)
    VELOCITIES_LIST = np.load(filename_vel)
    
    # --- Configure Plot Grid ---
    fig, ax = plt.subplots(4, 4, figsize=(12, 8))

    # Calculate stride: We want exactly 16 snapshots spanning the full simulation.
    plot_every_n_steps = int(POSITIONS_LIST.shape[1] / 15) # Aim for 16 plots
    if plot_every_n_steps == 0:
        plot_every_n_steps = 1

    for i in range(16):
        try:
            idx = i*plot_every_n_steps

            # --- Extract Data ---
            # Index 0: Primary Star
            # Index 1: Secondary Star (Perturber)
            # Index 2+: Disk Particles
            star1 = POSITIONS_LIST[0,idx,:]
            star2 = POSITIONS_LIST[1,idx,:]
            disk = POSITIONS_LIST[2:,idx,:]
            disk_vel = VELOCITIES_LIST[2:,idx,:]

            # Calculate speed scalar for coloring (hot/cold particles)
            disk_speed = np.linalg.norm(disk_vel, axis=1)

        except IndexError:
            print(f"IndexError at index {i} with stepsize {plot_every_n_steps}")
            continue
        
        # Map the linear loop index 'i' to the grid coordinates (row, col)
        a = int(i/4)
        b = i % 4

        # --- Center on Star 1 ---
        x0 = star1[0]
        y0 = star1[1]

        # --- Draw Scatter Plots ---
        # 1. Disk particles (colored by speed)
        sc = ax[a,b].scatter(disk[:,0] - x0, disk[:,1] - y0, c=disk_speed, cmap="hot", s=1, alpha=1)
        # 2. Star 1 (Blue, fixed at center)
        ax[a,b].scatter(0, 0, c='blue', s=100)
        # 3. Star 2 (Green, relative position)
        ax[a,b].scatter(star2[0] - x0, star2[1] - y0, c='green', s=100)
        
        # --- Formatting ---
        lim = 350
        ax[a,b].set_xlim(-lim, lim)
        ax[a,b].set_ylim(-lim, lim)
        ax[a,b].set_xlabel("x [AU]")
        ax[a,b].set_ylabel("y [AU]")
        ax[a,b].set_title(f"{i * plot_every_n_steps}yr")

    # --- Finalize Layout ---
    plt.tight_layout()
    cbar = fig.colorbar(sc, ax=ax.ravel().tolist(), shrink=0.95)
    cbar.set_label('Velocity Magnitude [km/s]')

    # --- Save Figure ---
    s1 = filename_pos.split("__")[1]
    fig.savefig(f"PLOT/{PlotName}__{s1[:-4]}.png")
    #plt.show()
    plt.close(fig)


# --- Initial Conditions ---
def get_initial_values_new(arr):
    """
    Extracts and computes initial kinematic properties for a stellar encounter from a data row.

    This function parses a single row (pandas Series or dict-like object) containing 
    raw coordinate and velocity data for two interacting particles. It computes the 
    relative position and velocity vectors of the second particle with respect to 
    the first, assigns physical units (AMUSE units), and derives stellar radii 
    based on mass.

    Parameters
    ----------
    arr : pandas.Series or dict
        A row containing the following keys (column names):
        - 'particle1_id', 'particle2_id'
        - 'time (yr)'
        - 'particle1_x (kpc)', 'particle1_y (kpc)', 'particle1_z (kpc)'
        - 'particle2_x (kpc)', 'particle2_y (kpc)', 'particle2_z (kpc)'
        - 'particle1_vx (km/s)', 'particle1_vy (km/s)', 'particle1_vz (km/s)'
        - 'particle2_vx (km/s)', 'particle2_vy (km/s)', 'particle2_vz (km/s)'
        - 'particle1_mass (MSun)', 'particle2_mass (MSun)'

    Returns
    -------
    tuple
        A tuple containing the following elements:
        - star1_id (int/str): ID of the primary star.
        - star2_id (int/str): ID of the secondary star.
        - mass_1 (quantity): Mass of the primary star (with units.MSun).
        - mass_2 (quantity): Mass of the secondary star (with units.MSun).
        - radius_1 (quantity): Radius of the primary star (derived from mass).
        - radius_2 (quantity): Radius of the secondary star (derived from mass).
        - time (quantity): The timestamp of the interaction (with units.yr).
        - rel_dist_vec (quantity): 3D vector [x, y, z] of the secondary star's 
          position relative to the primary, in units.kpc.
        - rel_vel_vec (quantity): 3D vector [vx, vy, vz] of the secondary star's 
          velocity relative to the primary, in units.kms.

    Notes
    -----
    - Uses `get_radius_from_mass` to determine stellar radii. Ensure this helper 
      function is defined in the scope.
    - All output vectors are in the frame where Particle 1 is at the origin (0,0,0).
    """

    # --- Extract Identifiers and Time ---
    star1_id = arr["particle1_id"]            
    star2_id = arr["particle2_id"]
    time = arr["time (yr)"] | units.yr

    # --- Extract Raw Coordinates (Particle 1) ---
    p1_x_kpc = arr["particle1_x (kpc)"]
    p1_y_kpc = arr["particle1_y (kpc)"]
    p1_z_kpc = arr["particle1_z (kpc)"]

    p1_vx_kms = arr["particle1_vx (km/s)"]
    p1_vy_kms = arr["particle1_vy (km/s)"]
    p1_vz_kms = arr["particle1_vz (km/s)"]
    
    # --- Extract Raw Coordinates (Particle 2) ---
    p2_x_kpc = arr["particle2_x (kpc)"]
    p2_y_kpc = arr["particle2_y (kpc)"]
    p2_z_kpc = arr["particle2_z (kpc)"]

    p2_vx_kms = arr["particle2_vx (km/s)"]
    p2_vy_kms = arr["particle2_vy (km/s)"]
    p2_vz_kms = arr["particle2_vz (km/s)"]

    # --- Compute Relative Frames ---
    rel_dx_kpc = p2_x_kpc - p1_x_kpc
    rel_dy_kpc = p2_y_kpc - p1_y_kpc
    rel_dz_kpc = p2_z_kpc - p1_z_kpc

    rel_dvx_kms = p2_vx_kms - p1_vx_kms
    rel_dvy_kms = p2_vy_kms - p1_vy_kms
    rel_dvz_kms = p2_vz_kms - p1_vz_kms
   
    # --- Assign units ---
    mass_1 = arr["particle1_mass (MSun)"] | units.MSun
    mass_2 = arr["particle2_mass (MSun)"] | units.MSun
     
    rel_dist_vec = [rel_dx_kpc, rel_dy_kpc, rel_dz_kpc] | units.kpc
    rel_vel_vec = [rel_dvx_kms, rel_dvy_kms, rel_dvz_kms] | units.kms

    # --- Derive Radii ---
    radius_1 = get_radius_from_mass(mass_1)     # 1 | units.RSun
    radius_2 = get_radius_from_mass(mass_2)     # 1 | units.RSun

    return star1_id, star2_id, mass_1, mass_2, radius_1, radius_2, time, rel_dist_vec, rel_vel_vec


def get_radius_from_mass(mass):
    """
    Estimates the radius of a Main Sequence star based on its mass using 
    standard empirical power-law relations.

    This function applies a piecewise approximation for the Mass-Radius relation:
    - For stars M > 1 MSun: R ~ M^0.8
    - For stars M <= 1 MSun: R ~ M^0.57 (approximating the lower main sequence)

    Parameters
    ----------
    mass : scalar(units.MSun)
        The mass of the star.

    Returns
    -------
    scalar(units.RSun)
        The estimated radius of the star.
    """
    Ms = 1 | units.MSun
    if mass > Ms:
        return (mass / Ms)**0.8 | units.RSun
    else:
        return (mass / Ms)**0.57 | units.RSun


# --- Saving + Loading Disks
def save_disk(disk, filename):
    '''
    Careful: Causes problems if subfolder for saving doesnt already exist or if the file already exists
    
    :param disk: the disk to be saved
    :param filename: where to save the disk
    '''
    write_set_to_file(disk, filename, "amuse", overwrite_file=True)


def load_disk(filename):
    """
    Loads a saved AMUSE particle set from a file and separates it into a central 
    star and a protoplanetary disk based on mass.

    This function assumes a specific mass hierarchy: particles with mass >= 1 MSun 
    are treated as stars, and particles < 1 MSun are treated as disk components.
    It is typically used to resume simulations or analyze saved states.

    Parameters
    ----------
    filename : str
        The path to the AMUSE data file (usually .amuse format) containing 
        the particle set.

    Returns
    -------
    tuple (amuse.datamodel.particles.Particle, amuse.datamodel.particles.Particles)
        - The first element is the primary Star particle (the first massive object found).
        - The second element is a Particles set containing all Disk components.
    """

    # Load the entire particle set (Star + Disk) from the AMUSE file
    all_particles = read_set_from_file(filename, "amuse")
    # --- Separation Logic ---
    # Create boolean masks to separate massive objects (stars) from light objects (disk)
    star_mask = (all_particles.mass >= 1 | units.MSun)
    disk_mask = (all_particles.mass < 1 | units.MSun)
    # Apply masks to create distinct subsets
    disk = all_particles[disk_mask]
    stars = all_particles[star_mask]
    # Return the primary star and the entire disk set
    return stars[0], disk


# --- Examples for Usage ---
if __name__ == "__main__":
    #print("ATTENTION: The filenames and IDs will have to be adapted to existing files!")

    # --- run all sims from interaction file ---
    #run_all_cluster_sims("Cluster_velocity_0_good.csv")

    # --- run sim with disks around both stars for stars specified
    #run_sim_2disk("Interactions stopping conditions_100Myr.csv", 113871276108901526, 15793569915568245741, t_sim=500|units.yr)
    
    # --- make animation from the data files provided
    fp = "Data/FullRunNewPosAU__100448.652139Myr_40_0_1000_1.076_9.331.npy"
    fv = "Data/FullRunNewVelKMS__100448.652139Myr_40_0_1000_1.076_9.331.npy"
    load_and_animate_data(fp, fv)


