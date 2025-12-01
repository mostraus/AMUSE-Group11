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

from functions import get_bound_particles_fraction, append_row_to_csv, write_bound_frac


def get_radius_from_mass(mass):
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


def simulate_two_hydro_disks(filename, main_character_star_idx, partner_star_idx, index=0):
    # --- 0. Get values from file ---
    arr = get_first_interaction(filename, main_character_star_idx, partner_star_idx, index)
    S1id, S2id, M1, M2, R1, R2, T, REL_DIST, REL_VEL = get_initial_values(arr)
    # --- 1. Create the Star ---
    #Mstar = 1.0 | units.MSun
    print(REL_DIST, REL_VEL)
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


def simulate_hydro_disk(filename, main_character_star_idx, partner_star_idx, give_Disk=None, index=0, t_sim=500 | units.yr):
    # --- 0. Get values from file ---
    arr = get_first_interaction(filename, main_character_star_idx, partner_star_idx, index)
    S1id, S2id, M1, M2, R1, R2, T, REL_DIST, REL_VEL = get_initial_values(arr)
    # --- 1. Create the Star ---
    #Mstar = 1.0 | units.MSun
    print(REL_DIST, REL_VEL)
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
    Ndisk = 2000 # Increased for better visuals
    Mdisk = 0.01 | units.MSun
    Rmin = 1.0 | units.au
    Rmax = 100.0 | units.au

    # Converter for the hydro code (scaled to disk properties)
    hydro_converter = nbody_system.nbody_to_si(M1, Rmax)
    
    if give_Disk:
        disk = give_Disk

    else:
        disk = ProtoPlanetaryDisk(Ndisk, 
                                convert_nbody=hydro_converter, 
                                Rmin=Rmin/Rmax, 
                                radius_min= Rmin/Rmax,
                                Rmax=1, 
                                radius_max= 1,
                                q_out=-1.5, # More typical surface density profile
                                discfraction=Mdisk/M1).result
        
        # The star is at (0,0), so no position/velocity offset is needed for the disk

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

    # --- 6. Setup Bridge ---
    gravhydro = bridge.Bridge(use_threading=False) # use_threading=False is often more stable
    gravhydro.add_system(gravity, (hydro,)) # Gravity (stars) acts on Hydro (disk)
    gravhydro.add_system(hydro, (gravity,)) # Hydro (disk) acts on Gravity (stars)
    
    dt = 1 | units.yr
    gravhydro.timestep = 0.1*dt # Bridge timestep

    # --- 7. Evolution Loop ---
    model_time = 0 | units.yr
    t_end = t_sim  # already has units from above
    
    times = np.linspace(0, t_end.value_in(units.yr), int(t_end/dt) + 1)

    POSITIONS_LIST = np.zeros((Ndisk + 2, times.shape[0], 3))
    VELOCITIES_LIST = np.zeros((Ndisk + 2, times.shape[0], 3))


    for i,t in enumerate(times):
        model_time = t | units.yr
        gravhydro.evolve_model(model_time)

        # Copy data back to particle sets
        ch2_disk.copy()
        ch2_stars.copy()
        
        print(f"t={model_time.in_(units.yr)}")
        POSITIONS_LIST[0][i][:] = np.array([stars[0].x.value_in(units.AU), stars[0].y.value_in(units.AU), stars[0].z.value_in(units.AU)])
        POSITIONS_LIST[1][i][:] = np.array([stars[1].x.value_in(units.AU), stars[1].y.value_in(units.AU), stars[1].z.value_in(units.AU)])
        VELOCITIES_LIST[0][i][:] = np.array([stars[0].vx.value_in(units.kms), stars[0].vy.value_in(units.kms), stars[0].vz.value_in(units.kms)])
        VELOCITIES_LIST[1][i][:] = np.array([stars[1].vx.value_in(units.kms), stars[1].vy.value_in(units.kms), stars[1].vz.value_in(units.kms)])
        for j in range(Ndisk):
            p = disk[j]
            POSITIONS_LIST[j+2][i][:] = np.array([p.x.value_in(units.AU), p.y.value_in(units.AU), p.z.value_in(units.AU)])
            VELOCITIES_LIST[j+2][i][:] = np.array([p.vx.value_in(units.kms), p.vy.value_in(units.kms), p.vz.value_in(units.kms)])


    info_array = np.array([T.value_in(units.Myr), S1id, S2id, t_end.value_in(units.yr), M1.value_in(units.MSun), M2.value_in(units.MSun)])    
    # Time of cluster [Myr], Star 1 [id], Star 2 [id], duration of disk [yr], Mass 1 [MSun], Mass 2 [MSun]
    ch2_disk.copy()
    save_disk(disk, f"DISK/DiskSave_{info_array[0]}Myr_{info_array[1]}_{info_array[2]}_{info_array[3]}yr_{info_array[4]}MSun_{info_array[5]}MSun.amuse")
    # --- 8. Cleanup ---
    gravity.stop()
    hydro.stop()

    write_bound_frac(M1, M2, POSITIONS_LIST, VELOCITIES_LIST, REL_DIST, times, Mdisk/M1)

    return POSITIONS_LIST, VELOCITIES_LIST, info_array


def load_and_plot_data(filename_pos, filename_vel, PlotName="DiskPlot"):
    # 1. Get the directory where THIS script is located
    #script_dir = Path(__file__).parent

    # 2. Create the full path to your data file
    #file_path_pos = script_dir / filename_pos
    file_path_pos = filename_pos
    #print(f"Successfully loaded from: {file_path_pos}")
    POSITIONS_LIST = np.load(file_path_pos)
    VELOCITIES_LIST = np.load(filename_vel)

    
    # Setup 3x4 plot grid
    fig, ax = plt.subplots(4, 4, figsize=(12, 8))
    j = 0 # Plot index
    plot_every_n_steps = int(POSITIONS_LIST.shape[1] / 15) # Aim for 16 plots
    if plot_every_n_steps == 0:
        plot_every_n_steps = 1

    for i in range(16):
        try:
            idx = i*plot_every_n_steps
            star1 = POSITIONS_LIST[0,idx,:]
            star2 = POSITIONS_LIST[1,idx,:]
            disk = POSITIONS_LIST[2:,idx,:]
            disk_vel = VELOCITIES_LIST[2:,idx,:]
            disk_speed = np.linalg.norm(disk_vel, axis=1)
        except IndexError:
            print(f"IndexError at index {i} with stepsize {plot_every_n_steps}")
            continue

        a = int(i/4)
        b = i % 4
        x0 = star1[0]
        y0 = star1[1]
        sc = ax[a,b].scatter(disk[:,0] - x0, disk[:,1] - y0, c=disk_speed, cmap="hot", s=1, alpha=1)
        ax[a,b].scatter(0, 0, c='blue', s=100)
        ax[a,b].scatter(star2[0] - x0, star2[1] - y0, c='green', s=100)
        lim = 350
        ax[a,b].set_xlim(-lim, lim)
        ax[a,b].set_ylim(-lim, lim)
        ax[a,b].set_xlabel("x [AU]")
        ax[a,b].set_ylabel("y [AU]")
        ax[a,b].set_title(f"{i * plot_every_n_steps}Myr")

    plt.tight_layout()
    cbar = fig.colorbar(sc, ax=ax.ravel().tolist(), shrink=0.95)
    cbar.set_label('Velocity Magnitude [km/s]')
    s = filename_pos.split("_")
    try:
        fig.suptitle(f"Interaction between Stars {s[2]} and {s[3]} at Cluster Time: {s[1]}")
    except IndexError:
        fig.suptitle("Ugh, no title")
    fig.savefig(f"PLOT/{PlotName}{filename_pos[18:-4]}.png")
    #plt.show()


def load_and_animate_data(filename_pos, filename_vel):
    """
    Loads position data and creates an animation showing the evolution
    of all time steps.

    Args:
        filename_pos (str): Path to the NumPy file containing position data.
        filename_vel (str): Path to the NumPy file containing velocity data (unused in this function).
    """
    
    # --- 1. Load Data ---
    POSITIONS_LIST = np.load(filename_pos)
    VELOCITIES_LIST = np.load(filename_vel)
    
    # POSITIONS_LIST shape is assumed to be (N_particles, N_timesteps, 3)
    N_timesteps = POSITIONS_LIST.shape[1]
    
    # Separate data components
    star1_positions = POSITIONS_LIST[0, :, :]
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
    s = filename_pos.split("_")     # s[0] = StartString, s[1] = cluster time, s[2] = star 1, s[3] = star 2
    ax.set_title(f"Interaction between Stars {s[2]} and {s[3]} at Cluster Time: {s[1]}")
    
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


    # --- 3. Animation Update Function ---
    def update_frame(i):
        """
        Updates the plot elements for frame i.
        """
        # Get the position of Star 1 at the current time step i
        x0, y0, _ = star1_positions[i]
        
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
        
        # Star 1 is always at (0, 0) in the relative frame
        star2_plot.set_offsets(np.column_stack([star2_x_rel, star2_y_rel]))
        star1_plot.set_offsets(np.column_stack([0, 0]))

        # Update colors
        current_speeds = disk_speeds_all[:, i]
        disk_plot.set_array(current_speeds)

        # Update the time step text
        time_text.set_text(f"Time Step: {i} / {N_timesteps - 1}")
        
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
    disk = read_set_from_file(filename, "amuse")
    return disk



def run_sim():
    filename = "interactions.csv"
    main_character_star_idx = 10
    partner_star_idx = 38
    index = 2
    POSITIONS_LIST, VELOCITIES_LIST, info_array = simulate_hydro_disk(filename, main_character_star_idx, partner_star_idx, index=index)
    filename_pos = f"Data/DiskDataPosAU_{info_array[0]}Myr_{info_array[1]}_{info_array[2]}_{info_array[3]}yr_{info_array[4]}MSun_{info_array[5]}MSun.npy"
    filename_vel = f"Data/DiskDataVelKMS_{info_array[0]}Myr_{info_array[1]}_{info_array[2]}_{info_array[3]}yr_{info_array[4]}MSun_{info_array[5]}MSun.npy"
    np.save(filename_pos, POSITIONS_LIST)
    np.save(filename_vel, VELOCITIES_LIST)
    return filename_pos, filename_vel


def run_sim_multiple_encounters():
    filename = "interactions.csv"
    main_character_star_idx = 0
    partner_star_idx = 51
    POSITIONS_LIST, VELOCITIES_LIST, info_array = simulate_hydro_disk(filename, main_character_star_idx, partner_star_idx)
    filename_pos = f"Data/DiskDataPosAU_{info_array[0]}Myr_{info_array[1]}_{info_array[2]}_{info_array[3]}yr_{info_array[4]}MSun_{info_array[5]}MSun.npy"
    filename_vel = f"Data/DiskDataVelKMS_{info_array[0]}Myr_{info_array[1]}_{info_array[2]}_{info_array[3]}yr_{info_array[4]}MSun_{info_array[5]}MSun.npy"
    np.save(filename_pos, POSITIONS_LIST)
    np.save(filename_vel, VELOCITIES_LIST)
    load_and_plot_data(filename_pos, filename_vel)
    plt.close()
    load_and_animate_data(filename_pos, filename_vel)
    filename_disk = f"DISK/DiskSave_{info_array[0]}Myr_{info_array[1]}_{info_array[2]}_{info_array[3]}yr_{info_array[4]}MSun_{info_array[5]}MSun.amuse"
    POSITIONS_LIST, VELOCITIES_LIST, info_array = simulate_hydro_disk(filename, main_character_star_idx, partner_star_idx, load_disk(filename_disk), index=1)
    filename_pos = f"Data/DiskDataPosAU_{info_array[0]}Myr_{info_array[1]}_{info_array[2]}_{info_array[3]}yr_{info_array[4]}MSun_{info_array[5]}MSun.npy"
    filename_vel = f"Data/DiskDataVelKMS_{info_array[0]}Myr_{info_array[1]}_{info_array[2]}_{info_array[3]}yr_{info_array[4]}MSun_{info_array[5]}MSun.npy"
    np.save(filename_pos, POSITIONS_LIST)
    np.save(filename_vel, VELOCITIES_LIST)
    load_and_plot_data(filename_pos, filename_vel)
    plt.close()
    load_and_animate_data(filename_pos, filename_vel)


def main():
    #filename_pos = "Data/DiskDataPosAU_43.0 MyrMyr_29_61_1000yr_10.0792049182MSun_7.75182634499MSun.npy" #"Data/DiskDataPosAU.npy"    #"DiskDataPosAU_7.0 MyrMyr_18_53_1000yr_1.20301815447MSun_3.07138236035MSun.npy"
    #filename_vel = "Data/DiskDataVelKMS_43.0 MyrMyr_29_61_1000yr_10.0792049182MSun_7.75182634499MSun.npy" #"Data/DiskDataVelKMS.npy"    #"DiskDataVelKMS_7.0 MyrMyr_18_53_1000yr_1.20301815447MSun_3.07138236035MSun.npy"
    filename_pos, filename_vel = run_sim()
    load_and_plot_data(filename_pos, filename_vel)
    load_and_animate_data(filename_pos, filename_vel)


if __name__ in ('__main__'):
    main()
    #run_sim_multiple_encounters()

    
    
