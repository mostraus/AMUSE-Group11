from amuse.lab import *
import numpy as np
from matplotlib import pyplot as plt
from amuse.couple import bridge
# from amuse.ext.orbital_elements import orbital_elements_from_binary # No longer needed
from amuse.ext.protodisk import ProtoPlanetaryDisk
import os.path
import csv
import pandas as pd
from pathlib import Path


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
    star1_id = interactions[interaction_id]["star_i"]            # mass and radius are sufficient
    star2_id = interactions[interaction_id]["star_j"]            # mass and radius are sufficient
    time_id = interactions[interaction_id]["time_index"]         # should be absolute time
    rel_dist = interactions[interaction_id]["distance_pc"]       # should be a vector with rel_x, rel_y, rel_z
    rel_vel = interactions[interaction_id]["rel_velocity_kms"]   # should also be a vector with rel_vx, rel_vy, rel_vz

    ########### THESE VALUES ONLY FOR NOW UNTIL FILE IS ADJUSTED ###########
    mass_1 = 1 | units.MSun
    mass_2 = 2 | units.MSun
    radius_1 = 1 | units.RSun
    radius_2 = 1 | units.RSun
    time = 0 | units.Myr
    rel_dist_vec = [-200, 100, 0] | units.AU
    rel_vel_vec = [5, 0, 0] | units.kms
    ########################################################################

    return mass_1, mass_2, radius_1, radius_2, time, rel_dist_vec, rel_vel_vec


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
    M1, M2, R1, R2, T, REL_DIST, REL_VEL = get_initial_values(filename, main_character_star_idx)
    # --- 1. Create the Star ---
    #Mstar = 1.0 | units.MSun
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

    return POSITIONS_LIST, VELOCITIES_LIST


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
        ax[a,b].set_xlim(-250, 250)
        ax[a,b].set_ylim(-250, 250)
        ax[a,b].set_xlabel("x [AU]")
        ax[a,b].set_ylabel("y [AU]")


    plt.tight_layout()
    fig.savefig("Plot4.png")
    plt.show()


def run_sim():
    filename = "interactions.csv"
    main_character_star_idx = 11
    POSITIONS_LIST, VELOCITIES_LIST = simulate_hydro_disk(filename, main_character_star_idx)
    np.save("DiskDataPosAU.npy", POSITIONS_LIST)
    np.save("DiskDataVelKMS.npy", VELOCITIES_LIST)

def main():
    run_sim()
    load_and_plot_data("DiskDataPosAU.npy", "DiskDataVelKMS.npy")

if __name__ in ('__main__'):
    main()
    
    
