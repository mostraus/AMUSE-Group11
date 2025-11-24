from amuse.lab import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from amuse.couple import bridge
# from amuse.ext.orbital_elements import orbital_elements_from_binary # No longer needed
from amuse.ext.protodisk import ProtoPlanetaryDisk
import os.path
import csv
import pandas as pd
from pathlib import Path


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

    # 3. Access rows 1 by 1
    # Example: Get all entries for a specific star (e.g., "Proxima")
    #    if star_id in stars_dict:
    #        rows = stars_dict[star_id]
    #        for row in rows:
    #            # You can now access values by column name
    #            t = row['time_index']
    #            dist = row['distance_pc']
    #            print(f"Time: {t}, Distance: {dist}")


def get_initial_values(filename, main_character_star_idx, interaction_id=0):
    stars_dict = read_interactions_file(filename)
    interactions = stars_dict[main_character_star_idx]
    arr = interactions[interaction_id]
    star1_id = arr["star_i"]            
    star2_id = arr["star_j"]
    time = arr["time "] | units.Myr
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


def simulate_hydro_disk(filename, main_character_star_idx):
    # --- 0. Get values from file ---
    S1id, S2id, M1, M2, R1, R2, T, REL_DIST, REL_VEL = get_initial_values(filename, main_character_star_idx)
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
    Ndisk = 4000 # Increased for better visuals
    Mdisk = 0.01 | units.MSun
    Rmin = 1.0 | units.au
    Rmax = 100.0 | units.au

    # Converter for the hydro code (scaled to disk properties)
    hydro_converter = nbody_system.nbody_to_si(M1, Rmax)
    
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
    
    gravity = Hermite(gravity_converter)
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
    t_end = 1000 | units.yr # Simulate for 200 years to see the fly-by
    
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

    # --- 8. Cleanup ---
    gravity.stop()
    hydro.stop()

    info_array = np.array([T.value_in(units.Myr), S1id, S2id, t_end.value_in(units.yr), M1.value_in(units.MSun), M2.value_in(units.MSun)])    
    # Time of cluster [Myr], Star 1 [id], Star 2 [id], duration of disk [yr], Mass 1 [MSun], Mass 2 [MSun]

    return POSITIONS_LIST, VELOCITIES_LIST, info_array


def load_and_plot_data(filename_pos, filename_vel):
    # 1. Get the directory where THIS script is located
    #script_dir = Path(__file__).parent

    # 2. Create the full path to your data file
    #file_path_pos = script_dir / filename_pos
    file_path_pos = filename_pos
    #print(f"Successfully loaded from: {file_path_pos}")
    POSITIONS_LIST = np.load(file_path_pos)
    #VELOCITIES_LIST = np.load(filename_vel)

    
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
        except IndexError:
            print(f"IndexError at index {i} with stepsize {plot_every_n_steps}")
        a = int(i/4)
        b = i % 4
        x0 = star1[0]
        y0 = star1[1]
        ax[a,b].scatter(disk[:,0] - x0, disk[:,1] - y0, c="black", s=1, alpha=0.3)
        ax[a,b].scatter(0, 0, c='red', s=100)
        ax[a,b].scatter(star2[0] - x0, star2[1] - y0, c='blue', s=100)
        lim = 350
        ax[a,b].set_xlim(-lim, lim)
        ax[a,b].set_ylim(-lim, lim)
        ax[a,b].set_xlabel("x [AU]")
        ax[a,b].set_ylabel("y [AU]")
        ax[a,b].set_title(f"{i * plot_every_n_steps}Myr")


    plt.tight_layout()
    fig.savefig(f"DiskPlot{filename_pos[13:-4]}.png")
    plt.show()


def load_and_animate_data(filename_pos, filename_vel):
    """
    Loads position data and creates an animation showing the evolution
    of all time steps.

    Args:
        filename_pos (str): Path to the NumPy file containing position data.
        filename_vel (str): Path to the NumPy file containing velocity data (unused in this function).
    """
    
    # --- 1. Load Data ---
    file_path_pos = filename_pos
    POSITIONS_LIST = np.load(file_path_pos)
    
    # POSITIONS_LIST shape is assumed to be (N_particles, N_timesteps, 3)
    N_timesteps = POSITIONS_LIST.shape[1]
    
    # Separate data components
    star1_positions = POSITIONS_LIST[0, :, :]
    star2_positions = POSITIONS_LIST[1, :, :]
    disk_positions = POSITIONS_LIST[2:, :, :]
    
    # --- 2. Setup Figure and Initial Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    lim = 300 # Set plot limits
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.set_title("Hydrodynamic Disk Evolution (Time Step: 0)")
    
    # Initialize the plot elements
    # Star 1 (Red, centered at 0,0 after subtraction)
    star1_plot = ax.scatter(0, 0, c='red', s=100) 
    
    # Star 2 (Blue, relative to Star 1)
    star2_plot = ax.scatter(0, 0, c='blue', s=100) 
    
    # Disk particles (Black/Gray)
    disk_plot = ax.scatter([], [], c="black", s=1, alpha=0.3) 
    
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
    output_filename = f"DiskAnimation{filename_pos[13:-4]}.mp4"
    print(f"Saving animation to {output_filename}...")
    
    # Use 'ffmpeg' writer for MP4. You might need to install it on your system.
    # Use 'imagemagick' writer for GIF (slower).
    
    anim.save(output_filename, writer='ffmpeg', fps=20) 

    print("Animation saved!")
    plt.close(fig) # Close the figure to free up memory
    
    return anim # Optionally return the animation object


def run_sim():
    filename = "interactions.csv"
    main_character_star_idx = 18
    POSITIONS_LIST, VELOCITIES_LIST, info_array = simulate_hydro_disk(filename, main_character_star_idx)
    filename_pos = f"DiskDataPosAU_{info_array[0]}Myr_{info_array[1]}_{info_array[2]}_{info_array[3]}yr_{info_array[4]}MSun_{info_array[5]}MSun.npy"
    filename_vel = f"DiskDataVelKMS_{info_array[0]}Myr_{info_array[1]}_{info_array[2]}_{info_array[3]}yr_{info_array[4]}MSun_{info_array[5]}MSun.npy"
    np.save(filename_pos, POSITIONS_LIST)
    np.save(filename_vel, VELOCITIES_LIST)
    return filename_pos, filename_vel


def main():
    filename_pos = "DiskDataPosAU.npy"    #"DiskDataPosAU_7.0 MyrMyr_18_53_1000yr_1.20301815447MSun_3.07138236035MSun.npy"
    filename_vel = "DiskDataVelKMS.npy"    #"DiskDataVelKMS_7.0 MyrMyr_18_53_1000yr_1.20301815447MSun_3.07138236035MSun.npy"
    #filename_pos, filename_vel = run_sim()
    load_and_plot_data(filename_pos, filename_vel)
    load_and_animate_data(filename_pos, filename_vel)
if __name__ in ('__main__'):
    main()
    
    
