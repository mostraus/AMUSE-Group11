### weird MacOS fix:
import os
# Allow MPI to oversubscribe (use more processes than physical cores)
os.environ["OMPI_MCA_rmaps_base_oversubscribe"] = "1"
# Use 'sh' instead of 'ssh' for local spawning (prevents ssh key errors)
os.environ["OMPI_MCA_plm_rsh_agent"] = "sh"
# Disable a specific memory mechanism that crashes on macOS
os.environ["OMPI_MCA_btl_vader_single_copy_mechanism"] = "none"

from amuse.lab import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from amuse.couple import bridge
# from amuse.ext.orbital_elements import orbital_elements_from_binary # No longer needed
from amuse.ext.protodisk import ProtoPlanetaryDisk
import csv
import pandas as pd
from pathlib import Path

from functions import get_bound_particles_fraction, write_bound_frac, calculate_energy, plot_energy_evolution


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


def read_interactions_file(filename="interactions.csv"):
    # 1. Load the dataset
    df = pd.read_csv(filename)

    # 2. Group by 'star_i' and create a dictionary
    # The result is a dict where:
    #   Key = star_i name
    #   Value = List of dictionaries (one for each row)
    stars_dict = {k: v.to_dict('records') for k, v in df.groupby('star_i')}
    return stars_dict


def get_first_interaction(filename, star_1, star_2=None, index=0):
    """
    Returns the first interaction for a specific star or a specific pair.
    
    Parameters:
    - df: The pandas DataFrame containing the interaction data.
    - star_1: The ID of the first star.
    - star_2: (Optional) The ID of the second star.
    
    Returns:
    - A pandas Series (row) representing the first interaction, or None if no interaction is found.
    """
    
    df = pd.read_csv(filename)
    # 1. Ensure data is sorted by time so .iloc[0] is actually the first event
    # If your CSV is already sorted, you can comment this line out to save time.
    #df_sorted = df.sort_values(by='time')
    df_sorted = df

    if star_2 is None:
        # CASE 1: Find first interaction of star_1 with ANYONE
        # We check if star_1 appears in EITHER the i or j column
        mask = (df_sorted['star_i'] == star_1) | (df_sorted['star_j'] == star_1)
    
    else:
        # CASE 2: Find first interaction between star_1 AND star_2
        # We must check both directions: (1 interacts with 2) OR (2 interacts with 1)
        mask = (
            ((df_sorted['star_i'] == star_1) & (df_sorted['star_j'] == star_2)) |
            ((df_sorted['star_i'] == star_2) & (df_sorted['star_j'] == star_1))
        )

    # Apply filter
    result = df_sorted[mask]

    # Return the first result or None if empty
    if result.empty:
        return None
    else:
        return result.iloc[index]


def get_interaction_new(filename, star_1, star_2, index=0):
    df = pd.read_csv(filename)
    mask = (df['particle1_id'] == star_1) & (df['particle2_id'] == star_2)
    result = df[mask]
    if result.empty:
        return None
    else:
        return result.iloc[index]


def get_initial_values(arr):

    star1_id = arr["star_i"]            
    star2_id = arr["star_j"]
    ### turn strings from csv into float values + units
    time = float(arr["time "][:-4]) | units.Myr
    rel_dx_pc = float(arr["relative_dx_pc"][1:-1])  # | units.pc
    rel_dy_pc = float(arr["relative_dy_pc"][1:-1])  # | units.pc
    rel_dz_pc = float(arr["relative_dz_pc"][1:-1])  # | units.pc
    rel_dvx_kms = float(arr["relative_dvx_kms"][1:-1])  # | units.kms
    rel_dvy_kms = float(arr["relative_dvy_kms"][1:-1])  # | units.kms
    rel_dvz_kms = float(arr["relative_dvz_kms"][1:-1])  # | units.kms
    #rel_dist = arr["distance_pc"]    
    mass_1 = arr["mass_star_i_MSun"] | units.MSun
    mass_2 = arr["mass_star_j_MSun"] | units.MSun
    #mass_ratio = arr["mass_ratio"]
     
    #rel_dist_vec = units.quantities.new_quantity([rel_dx_pc, rel_dy_pc, rel_dz_pc], units.pc) #| units.pc
    rel_dist_vec = [rel_dx_pc, rel_dy_pc, rel_dz_pc] | units.pc
    rel_vel_vec = [rel_dvx_kms, rel_dvy_kms, rel_dvz_kms] | units.kms
    ########### THESE VALUES ONLY FOR NOW UNTIL FILE IS ADJUSTED ###########
    #mass_1 = 1 | units.MSun
    #mass_2 = 20| units.MSun
    radius_1 = get_radius_from_mass(mass_1)     # 1 | units.RSun
    radius_2 = get_radius_from_mass(mass_2)     # 1 | units.RSun
    #time = 0 | units.Myr
    #rel_dist_vec = [-500, 300, 0] | units.AU
    #rel_vel_vec = [5, 0, 0] | units.kms
    ########################################################################

    return star1_id, star2_id, mass_1, mass_2, radius_1, radius_2, time, rel_dist_vec, rel_vel_vec


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


def plot_system_with_disk(stars, disk, model_time, ax):
    """
    Plots the star, the perturber, and the disk particles.
    """
    ax.set_title(f"t={model_time.value_in(units.yr):.2f}yr")
    
    star = stars[stars.name=="STAR"]
    perturber = stars[stars.name=="PERTURBER"]
    
    # Plot disk particles
    ax.scatter(disk.x.value_in(units.au) - star.x.value_in(units.au), disk.y.value_in(units.au) - star.y.value_in(units.au), c='k', s=1, alpha=0.3)
    
    # Plot star
    ax.scatter(0, 0, c='r', s=200, label="Star")
    
    # Plot perturber
    ax.scatter(perturber.x.value_in(units.au) - star.x.value_in(units.au), perturber.y.value_in(units.au) - star.x.value_in(units.au), c='b', s=150, label="Perturber")
    
    # Set fixed limits to observe the encounter
    lim = 250
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    #ax.set_aspect('equal')
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    #ax.legend(loc="upper right")


# TODO: apply changes made to "simulate hydro disk" to be up to date again
def simulate_two_hydro_disks(filename, main_character_star_idx, partner_star_idx, index=0):
    # --- 0. Get values from file ---
    arr = get_first_interaction(filename, main_character_star_idx, partner_star_idx, index)
    S1id, S2id, M1, M2, R1, R2, T, REL_DIST, REL_VEL = get_initial_values(arr)
    # --- 1. Create the Star ---
    star = Particles(1)
    star.mass = M1
    star.radius = R1 #1.0 | units.RSun
    star.position = (0, 0, 0) | units.au    # Rest frame of this star
    star.velocity = (0, 0, 0) | units.kms
    star.name = "STAR"

    # --- 2. Create the Perturbing Star ---
    # Placed 200 AU away with an impact parameter of 50 AU
    # Velocity of 10 km/s (~2.1 AU/yr) means encounter happens ~100 yr
    perturber = Particles(1)
    perturber.mass = M2 #0.5 | units.MSun
    perturber.radius = R2 #0.5 | units.RSun
    perturber.position = REL_DIST #(-200.0, 100.0, 0.0) | units.au
    perturber.velocity = REL_VEL #(5.0, 0.0, 0.0) | units.kms
    perturber.name = "PERTURBER"

    # Particle set for all massive N-body objects
    stars = Particles()
    stars.add_particle(star)
    stars.add_particle(perturber)

    # --- 3. Create the Gas Disk around the primary star ---
    Ndisk1 = 2000 # Increased for better visuals
    Mdisk1 = 0.01 | units.MSun
    Rmin1 = 1.0 | units.au
    Rmax1 = 100.0 | units.au

    # Converter for the hydro code (scaled to disk properties)
    hydro_converter = nbody_system.nbody_to_si(M1, Rmax1)

    disk = ProtoPlanetaryDisk(Ndisk1, 
                            convert_nbody=hydro_converter, 
                            Rmin=Rmin1/Rmax1, 
                            radius_min= Rmin1/Rmax1,
                            Rmax=1, 
                            radius_max= 1,
                            q_out=-1.5, # More typical surface density profile
                            discfraction=Mdisk1/M1).result
        
    # The star is at (0,0), so no position/velocity offset is needed for the disk

    # --- 3. Create the Gas Disk around the secondary star ---
    Ndisk2 = 2000 # Increased for better visuals
    Mdisk2 = 0.01 | units.MSun
    Rmin2 = 1.0 | units.au
    Rmax2 = 100.0 | units.au

    # Converter for the hydro code (scaled to disk properties)
    hydro_converter2 = nbody_system.nbody_to_si(M2, Rmax2)

    disk2 = ProtoPlanetaryDisk(Ndisk2, 
                            convert_nbody=hydro_converter2, 
                            Rmin=Rmin2/Rmax2, 
                            radius_min= Rmin2/Rmax2,
                            Rmax=1, 
                            radius_max= 1,
                            q_out=-1.5, # More typical surface density profile
                            discfraction=Mdisk2/M2).result
    
    # this disk needs a position and velocity boost to be around the second star
    disk2.position += REL_DIST
    disk2.velocity += REL_VEL

    # --- 4. Setup Gravity Code (for stars) ---
    # Converter for the N-body code (scaled to the stellar system)
    gravity_converter = nbody_system.nbody_to_si(stars.mass.sum(), 400.0 | units.au)
    
    gravity = Hermite(gravity_converter) #, channel_type="sockets")
    gravity.particles.add_particles(stars)
    ch2_stars = gravity.particles.new_channel_to(stars)

    # --- 5. Setup Hydro Code (for gas disk) ---
    hydro = Fi(hydro_converter, mode="openmp")
    hydro.parameters.timestep = 0.05 | units.yr # Adjusted for disk timescale
    hydro.particles.add_particles(disk)
    ch2_disk = hydro.particles.new_channel_to(disk)
    hydro.particles.add_particles(disk2)
    ch2_disk2 = hydro.particles.new_channel_to(disk2)

    # --- 6. Setup Bridge ---
    gravhydro = bridge.Bridge(use_threading=False) # use_threading=False is often more stable
    gravhydro.add_system(gravity, (hydro,)) # Gravity (stars) acts on Hydro (disk)
    gravhydro.add_system(hydro, (gravity,)) # Hydro (disk) acts on Gravity (stars)
    
    dt = 1 | units.yr
    gravhydro.timestep = 0.1*dt # Bridge timestep

    # --- 7. Evolution Loop ---
    model_time = 0 | units.yr
    t_end = 500 | units.yr # Simulate for 200 years to see the fly-by
    
    times = np.linspace(0, t_end.value_in(units.yr), int(t_end/dt) + 1)

    POSITIONS_LIST = np.zeros((Ndisk1 + Ndisk2 + 2, times.shape[0], 3))
    VELOCITIES_LIST = np.zeros((Ndisk1 + Ndisk2 + 2, times.shape[0], 3))


    for i,t in enumerate(times):
        model_time = t | units.yr
        gravhydro.evolve_model(model_time)

        # Copy data back to particle sets
        ch2_disk.copy()
        ch2_disk2.copy()
        ch2_stars.copy()
        
        print(f"t={model_time.in_(units.yr)}")
        POSITIONS_LIST[0][i][:] = np.array([stars[0].x.value_in(units.AU), stars[0].y.value_in(units.AU), stars[0].z.value_in(units.AU)])
        POSITIONS_LIST[1][i][:] = np.array([stars[1].x.value_in(units.AU), stars[1].y.value_in(units.AU), stars[1].z.value_in(units.AU)])
        VELOCITIES_LIST[0][i][:] = np.array([stars[0].vx.value_in(units.kms), stars[0].vy.value_in(units.kms), stars[0].vz.value_in(units.kms)])
        VELOCITIES_LIST[1][i][:] = np.array([stars[1].vx.value_in(units.kms), stars[1].vy.value_in(units.kms), stars[1].vz.value_in(units.kms)])
        for j in range(Ndisk1 + Ndisk2):
            if j < Ndisk1:
                p = disk[j]
            else:
                p = disk2[j - Ndisk1]
            POSITIONS_LIST[j+2][i][:] = np.array([p.x.value_in(units.AU), p.y.value_in(units.AU), p.z.value_in(units.AU)])
            VELOCITIES_LIST[j+2][i][:] = np.array([p.vx.value_in(units.kms), p.vy.value_in(units.kms), p.vz.value_in(units.kms)])


    info_array = np.array([T.value_in(units.Myr), S1id, S2id, t_end.value_in(units.yr), M1.value_in(units.MSun), M2.value_in(units.MSun)])    
    # Time of cluster [Myr], Star 1 [id], Star 2 [id], duration of disk [yr], Mass 1 [MSun], Mass 2 [MSun]
    #ch2_disk.copy()
    #ch2_disk2.copy()
    #save_disk(disk, f"DISK/DiskSave_{info_array[0]}Myr_{info_array[1]}_{info_array[2]}_{info_array[3]}yr_{info_array[4]}MSun_{info_array[5]}MSun.amuse")
    # --- 8. Cleanup ---
    gravity.stop()
    hydro.stop()

    return POSITIONS_LIST, VELOCITIES_LIST, info_array


def simulate_hydro_disk(filename, 
                        main_character_star_idx, 
                        partner_star_idx, 
                        disk_file=None, 
                        index=0,
                        t_sim=500 | units.yr, 
                        dt=1 | units.yr,
                        N_disk=2000, 
                        M_disk=0.01 | units.MSun, 
                        R_min=1.0 | units.au, 
                        R_max=100.0 | units.au, 
                        q_out=-1.5):
    # --- Get values from file ---
    #arr = get_first_interaction(filename, main_character_star_idx, partner_star_idx, index)
    #S1id, S2id, M1, M2, R1, R2, T, REL_DIST, REL_VEL = get_initial_values(arr)

    arr = get_interaction_new(filename, main_character_star_idx, partner_star_idx, index)
    print(arr)
    S1id, S2id, M1, M2, R1, R2, T, REL_DIST, REL_VEL = get_initial_values_new(arr)

    # --- Create the Perturbing Star ---
    perturber = Particles(1)
    perturber.mass = M2 
    perturber.radius = R2 
    perturber.position = REL_DIST   # Rest frame of other star
    perturber.velocity = REL_VEL 
    perturber.name = "PERTURBER"

    # Converter for the hydro code (scaled to disk properties)
    hydro_converter = nbody_system.nbody_to_si(M1, R_max)
    
    # --- Set up Disk ---
    if disk_file:
        star, disk = load_disk(disk_file)
        # bc pos and vel of perturber are relative to star and have to be modified to match its movement
        perturber.position += star.position
        perturber.velocity += star.velocity


    else:
        # --- Create the Star ---
        star = Particles(1)
        star.mass = M1
        star.radius = R1 
        star.position = (0, 0, 0) | units.au    # Rest frame of this star
        star.velocity = (0, 0, 0) | units.kms
        star.name = "STAR"

        disk = ProtoPlanetaryDisk(N_disk, 
                                convert_nbody=hydro_converter, 
                                Rmin=R_min/R_max, 
                                radius_min= R_min/R_max,
                                Rmax=1, 
                                radius_max= 1,
                                q_out=q_out, 
                                discfraction=M_disk/M1).result
        
        # The star is at (0,0), so no position/velocity offset is needed for the disk

    # Particle set for all massive N-body objects
    stars = Particles()
    stars.add_particle(star)
    stars.add_particle(perturber)

    # --- Setup Gravity Code (for stars) ---
    # Converter for the N-body code (scaled to the stellar system)
    gravity_converter = nbody_system.nbody_to_si(stars.mass.sum(), 400.0 | units.au)
    
    gravity = Hermite(gravity_converter) #, channel_type="sockets")
    gravity.particles.add_particles(stars)
    ch2_stars = gravity.particles.new_channel_to(stars)

    # --- Setup Hydro Code (for gas disk) ---
    hydro = Fi(hydro_converter, mode="openmp")
    hydro.parameters.timestep = 0.05 | units.yr # Adjusted for disk timescale
    hydro.particles.add_particles(disk)
    ch2_disk = hydro.particles.new_channel_to(disk)

    # --- Setup Bridge ---
    gravhydro = bridge.Bridge(use_threading=False) # use_threading=False is often more stable
    gravhydro.add_system(gravity, (hydro,)) # Gravity (stars) acts on Hydro (disk)
    gravhydro.add_system(hydro, (gravity,)) # Hydro (disk) acts on Gravity (stars)
    
    gravhydro.timestep = 0.1*dt # Bridge timestep

    # --- Simulation Setup ---
    model_time = 0 | units.yr
    t_end = t_sim  # already has units from above
    
    times = np.linspace(0, t_end.value_in(units.yr), int(t_end/dt) + 1)

    POSITIONS_LIST = np.zeros((N_disk + 2, times.shape[0], 3))
    VELOCITIES_LIST = np.zeros((N_disk + 2, times.shape[0], 3))

    # --- For the energy calculation ---
    all_particles = ParticlesSuperset([stars, disk])    # superset updates with its contents
    ENERGIES_J = np.zeros_like(times)

    # --- Simulation run ---
    for i,t in enumerate(times):
        model_time = t | units.yr
        gravhydro.evolve_model(model_time)

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
            p = disk[j]
            POSITIONS_LIST[j+2][i][:] = np.array([p.x.value_in(units.AU), p.y.value_in(units.AU), p.z.value_in(units.AU)])
            VELOCITIES_LIST[j+2][i][:] = np.array([p.vx.value_in(units.kms), p.vy.value_in(units.kms), p.vz.value_in(units.kms)])


    info_array = np.array([T.value_in(units.Myr), S1id, S2id, t_end.value_in(units.yr), M1.value_in(units.MSun), M2.value_in(units.MSun)])    
    # Time of cluster [Myr], Star 1 [id], Star 2 [id], duration of disk [yr], Mass 1 [MSun], Mass 2 [MSun]
    # --- Save the disk for further runs with the same disk ---
    ch2_disk.copy()
    ch2_stars.copy()
    save_str = f"{info_array[0]}Myr_{info_array[1]}_{info_array[2]}_{info_array[3]}yr_{info_array[4]}MSun_{info_array[5]}MSun"
    disk_savename = f"DISK/DiskSave_{save_str}.amuse"
    save_disk(all_particles, disk_savename)
    # --- Cleanup ---
    gravity.stop()
    hydro.stop()

    # --- Meta info extraction ---
    plot_energy_evolution(times, ENERGIES_J, f"PLOT/EnergyEvol_{save_str}.png")
    #write_bound_frac(M1, M2, POSITIONS_LIST, VELOCITIES_LIST, REL_DIST, times, M_disk/M1)

    return POSITIONS_LIST, VELOCITIES_LIST, info_array, disk_savename


def test_given_disk(ini_file=None, 
                    M1=1|units.MSun, 
                    M2=1|units.MSun, 
                    T=0|units.Myr, 
                    REL_DIST=[-200,200,0]|units.AU, 
                    REL_VEL=[5,0,0]|units.kms, 
                    t_sim=500 | units.yr, 
                    dt=1 | units.yr):

    S1id = -1
    S2id = -1
    R1 = get_radius_from_mass(M1)
    R2 = get_radius_from_mass(M2)

    # --- Create the Perturbing Star ---
    perturber = Particles(1)
    perturber.mass = M2 
    perturber.radius = R2 
    perturber.position = REL_DIST   # Rest frame of other star
    perturber.velocity = REL_VEL 
    perturber.name = "PERTURBER"

    # Converter for the hydro code (scaled to disk properties)
    R_max = 100 | units.AU
    R_min = 1 | units.AU
    N_disk = 2000
    q_out = -1.5
    M_disk = 0.01 | units.MSun
    hydro_converter = nbody_system.nbody_to_si(M1, R_max)

    # --- Set up Disk ---
    if ini_file:
        star, disk = load_disk(ini_file)
        # bc pos and vel of perturber are relative to star and have to be modified to match its movement
        perturber.position += star.position
        perturber.velocity += star.velocity

    else:
        # --- Create the Star ---
        star = Particles(1)
        star.mass = M1
        star.radius = R1 
        star.position = (0, 0, 0) | units.au    # Rest frame of this star
        star.velocity = (0, 0, 0) | units.kms
        star.name = "STAR"

        disk = ProtoPlanetaryDisk(N_disk, 
                                convert_nbody=hydro_converter, 
                                Rmin=R_min/R_max, 
                                radius_min= R_min/R_max,
                                Rmax=1, 
                                radius_max= 1,
                                q_out=q_out, 
                                discfraction=M_disk/M1).result

    # Particle set for all massive N-body objects
    stars = Particles()
    stars.add_particle(star)
    stars.add_particle(perturber)

    # --- Setup Gravity Code (for stars) ---
    # Converter for the N-body code (scaled to the stellar system)
    gravity_converter = nbody_system.nbody_to_si(stars.mass.sum(), 400.0 | units.au)
    
    gravity = Hermite(gravity_converter) #, channel_type="sockets")
    gravity.particles.add_particles(stars)
    ch2_stars = gravity.particles.new_channel_to(stars)

    # --- Setup Hydro Code (for gas disk) ---
    hydro = Fi(hydro_converter, mode="openmp")
    hydro.parameters.timestep = 0.05 | units.yr # Adjusted for disk timescale
    hydro.particles.add_particles(disk)
    ch2_disk = hydro.particles.new_channel_to(disk)

    # --- Setup Bridge ---
    gravhydro = bridge.Bridge(use_threading=False) # use_threading=False is often more stable
    gravhydro.add_system(gravity, (hydro,)) # Gravity (stars) acts on Hydro (disk)
    gravhydro.add_system(hydro, (gravity,)) # Hydro (disk) acts on Gravity (stars)
    
    gravhydro.timestep = 0.1*dt # Bridge timestep

    # --- Simulation Setup ---
    model_time = 0 | units.yr
    t_end = t_sim  # already has units from above
    
    times = np.linspace(0, t_end.value_in(units.yr), int(t_end/dt) + 1)
    POSITIONS_LIST = np.zeros((N_disk + 2, times.shape[0], 3))
    VELOCITIES_LIST = np.zeros((N_disk + 2, times.shape[0], 3))

    # --- For the energy calculation ---
    all_particles = ParticlesSuperset([stars, disk])    # superset updates with its contents
    ENERGIES_J = np.zeros_like(times)

    # --- Simulation run ---
    for i,t in enumerate(times):
        model_time = t | units.yr
        gravhydro.evolve_model(model_time)

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
            p = disk[j]
            POSITIONS_LIST[j+2][i][:] = np.array([p.x.value_in(units.AU), p.y.value_in(units.AU), p.z.value_in(units.AU)])
            VELOCITIES_LIST[j+2][i][:] = np.array([p.vx.value_in(units.kms), p.vy.value_in(units.kms), p.vz.value_in(units.kms)])


    info_array = np.array([T.value_in(units.Myr), S1id, S2id, t_end.value_in(units.yr), M1.value_in(units.MSun), M2.value_in(units.MSun)])    
    # Time of cluster [Myr], Star 1 [id], Star 2 [id], duration of disk [yr], Mass 1 [MSun], Mass 2 [MSun]
    # --- Save the disk for further runs with the same disk ---
    ch2_disk.copy()
    ch2_stars.copy()
    save_str = f"{info_array[0]}Myr_{info_array[1]}_{info_array[2]}_{info_array[3]}yr_{info_array[4]}MSun_{info_array[5]}MSun"
    disk_filename = f"DISK/DiskTest_{save_str}__{POSITIONS_LIST[0][-1][0]}__{POSITIONS_LIST[0][-1][1]}__{POSITIONS_LIST[0][-1][2]}.amuse"
    save_disk(all_particles, disk_filename)
    # --- Cleanup ---
    gravity.stop()
    hydro.stop()

    # --- Meta info extraction ---
    plot_energy_evolution(times, ENERGIES_J, f"PLOT/EnergyEvol_{save_str}.png")
    #write_bound_frac(M1, M2, POSITIONS_LIST, VELOCITIES_LIST, REL_DIST, times, M_disk/M1)

    return POSITIONS_LIST, VELOCITIES_LIST, info_array, disk_filename 


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
    ax.set_title(f"Interaction between Masses {s[-2]} and {s[-1][:-4]} at Cluster Time: {s[2]}")
    
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


def plot_disk(disk, save_name):
    disk.move_to_center()
    fig, ax = plt.subplots()
    ax.scatter(disk.x.value_in(units.AU), disk.y.value_in(units.AU), s=1, c="black")
    ax.scatter(0,0, s=10, c="red")
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.set_xlim(-350, 350)
    ax.set_ylim(-350, 350)
    fig.savefig(save_name)


def run_sim():
    filename = "interactions.csv"
    main_character_star_idx = 10
    partner_star_idx = 38
    index = 0
    POSITIONS_LIST, VELOCITIES_LIST, info_array = simulate_hydro_disk(filename, main_character_star_idx, partner_star_idx, index=index)
    filename_pos = f"Data/DiskDataPosAU_{info_array[0]}Myr_{info_array[1]}_{info_array[2]}_{info_array[3]}yr_{info_array[4]}MSun_{info_array[5]}MSun.npy"
    filename_vel = f"Data/DiskDataVelKMS_{info_array[0]}Myr_{info_array[1]}_{info_array[2]}_{info_array[3]}yr_{info_array[4]}MSun_{info_array[5]}MSun.npy"
    np.save(filename_pos, POSITIONS_LIST)
    np.save(filename_vel, VELOCITIES_LIST)
    return filename_pos, filename_vel


def run_sim_multiple_encounters():
    filename = "interactions.csv"
    main_character_star_idx = 16952461524525630702
    partner_star_idx = 1497372959794584352
    POSITIONS_LIST, VELOCITIES_LIST, info_array = simulate_hydro_disk(filename, main_character_star_idx, partner_star_idx)
    filename_pos = f"Data/TEST1DiskDataPosAU_{info_array[0]}Myr_{info_array[1]}_{info_array[2]}_{info_array[3]}yr_{info_array[4]}MSun_{info_array[5]}MSun.npy"
    filename_vel = f"Data/TEST1DiskDataVelKMS_{info_array[0]}Myr_{info_array[1]}_{info_array[2]}_{info_array[3]}yr_{info_array[4]}MSun_{info_array[5]}MSun.npy"
    np.save(filename_pos, POSITIONS_LIST)
    np.save(filename_vel, VELOCITIES_LIST)
    load_and_plot_data(filename_pos, filename_vel)
    plt.close()
    load_and_animate_data(filename_pos, filename_vel)
    filename_disk = f"DISK/TEST1DiskSave_{info_array[0]}Myr_{info_array[1]}_{info_array[2]}_{info_array[3]}yr_{info_array[4]}MSun_{info_array[5]}MSun.amuse"
    POSITIONS_LIST, VELOCITIES_LIST, info_array = simulate_hydro_disk(filename, main_character_star_idx, partner_star_idx, load_disk(filename_disk), index=1)
    filename_pos = f"Data/TEST2DiskDataPosAU_{info_array[0]}Myr_{info_array[1]}_{info_array[2]}_{info_array[3]}yr_{info_array[4]}MSun_{info_array[5]}MSun.npy"
    filename_vel = f"Data/TEST2DiskDataVelKMS_{info_array[0]}Myr_{info_array[1]}_{info_array[2]}_{info_array[3]}yr_{info_array[4]}MSun_{info_array[5]}MSun.npy"
    np.save(filename_pos, POSITIONS_LIST)
    np.save(filename_vel, VELOCITIES_LIST)
    load_and_plot_data(filename_pos, filename_vel)
    plt.close()
    load_and_animate_data(filename_pos, filename_vel)


def disentangle_disk_filename(disk_filename):
    df = disk_filename.split("__")
    pos_1 = [float(df[-3].split(".")[0]), float(df[-2]), float(df[-1])] | units.AU
    df1 = df[0].split("_")
    directory, savename = df1[0].split("/")
    T, S1idx, S2idx, length, M1, M2 = df1[1:]
    T = float(T[:-3]) | units.Myr
    S1idx = float(S1idx)
    S2idx = float(S2idx)
    length = float(length[:-2]) | units.yr
    M1 = float(M1[:-4]) | units.MSun
    M2 = float(M2[:-4]) | units.MSun

    return pos_1, directory, savename, T, length, S1idx, S2idx, M1, M2


def run_test_disk():
    POSITIONS_LIST, VELOCITIES_LIST, info_array, disk_filename = test_given_disk(t_sim=300|units.yr)
    i=1
    filename_pos1 = f"Data/DiskTest{i}PosAU_{info_array[0]}Myr_{info_array[1]}_{info_array[2]}_{info_array[3]}yr_{info_array[4]}MSun_{info_array[5]}MSun.npy"
    filename_vel1 = f"Data/DiskTest{i}VelKMS_{info_array[0]}Myr_{info_array[1]}_{info_array[2]}_{info_array[3]}yr_{info_array[4]}MSun_{info_array[5]}MSun.npy"
    np.save(filename_pos1, POSITIONS_LIST)
    np.save(filename_vel1, VELOCITIES_LIST)
    #disk = load_disk(disk_filename)
    #print(disk_filename)
    #df = disk_filename.split("/")[1]
    #df = df.split(".")[0]
    #plot_disk(disk, f"PLOT/{df}.png")
    #pos1 = POSITIONS_LIST[0][-1][:] | units.AU
    POSITIONS_LIST, VELOCITIES_LIST, info_array, disk_filename = test_given_disk(ini_file=disk_filename, t_sim=1000|units.yr, M2=20|units.MSun)
    i=2
    filename_pos2 = f"Data/DiskTest{i}PosAU_{info_array[0]}Myr_{info_array[1]}_{info_array[2]}_{info_array[3]}yr_{info_array[4]}MSun_{info_array[5]}MSun.npy"
    filename_vel2 = f"Data/DiskTest{i}VelKMS_{info_array[0]}Myr_{info_array[1]}_{info_array[2]}_{info_array[3]}yr_{info_array[4]}MSun_{info_array[5]}MSun.npy"
    np.save(filename_pos2, POSITIONS_LIST)
    np.save(filename_vel2, VELOCITIES_LIST)

    load_and_plot_data(filename_pos1, filename_vel1, "DiskTest1")
    load_and_plot_data(filename_pos2, filename_vel2, "DiskTest2")


def main():
    #filename_pos = "Data/DiskDataPosAU_43.0 MyrMyr_29_61_1000yr_10.0792049182MSun_7.75182634499MSun.npy" #"Data/DiskDataPosAU.npy"    #"DiskDataPosAU_7.0 MyrMyr_18_53_1000yr_1.20301815447MSun_3.07138236035MSun.npy"
    #filename_vel = "Data/DiskDataVelKMS_43.0 MyrMyr_29_61_1000yr_10.0792049182MSun_7.75182634499MSun.npy" #"Data/DiskDataVelKMS.npy"    #"DiskDataVelKMS_7.0 MyrMyr_18_53_1000yr_1.20301815447MSun_3.07138236035MSun.npy"
    filename_pos, filename_vel = run_sim()
    load_and_plot_data(filename_pos, filename_vel)
    load_and_animate_data(filename_pos, filename_vel)


if __name__ in ('__main__'):
    #main()
    #run_sim_multiple_encounters()
    #fp1 = "Data/DiskTest1PosAU_0Myr_-1_-1_300yr_1MSun_1MSun.npy"
    #fv1 = "Data/DiskTest1VelKMS_0Myr_-1_-1_300yr_1MSun_1MSun.npy"
    #fp2 = "Data/DiskTest2PosAU_0Myr_-1_-1_1000yr_1MSun_20MSun.npy"
    #fv2 = "Data/DiskTest2VelKMS_0Myr_-1_-1_1000yr_1MSun_20MSun.npy"
    #load_and_plot_data(fp, fv)
    #load_and_animate_data(fp2, fv2)
    #run_sim()
    #run_test_disk()
    #disk = load_disk("DISK/DiskTest_0Myr_-1_-1_500yr_1MSun_1MSun__105.34723597382165__203.50617988729877__0.01024626570596548.amuse")
    #print(disk.center_of_mass().value_in(units.AU))
    #disk.move_to_center()
    #print(disk.center_of_mass().value_in(units.AU))
    fp = "Data/FullRunNewPosAU__100448.652139Myr_40_0_1000_1.076_9.331.npy"
    fv = "Data/FullRunNewVelKMS__100448.652139Myr_40_0_1000_1.076_9.331.npy"
    load_and_plot_data(fp, fv, PlotName="ReportPlot")


    
    


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
