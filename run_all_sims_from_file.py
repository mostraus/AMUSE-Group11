import pandas as pd
import numpy as np
from amuse.lab import *
from amuse.ext.protodisk import ProtoPlanetaryDisk
from HydroExample2 import simulate_hydro_disk, load_and_plot_data, load_and_animate_data, get_initial_values_new, simulate_disk_new

# TODO: make sure saving the data and plots works correctly
# TODO: naming the stars nicer in plots and data
# TODO: check why the disk implodes
# TODO: clean up functions


def marker_sizes(rel_z_distances, max_size=150, min_size=10):
    abs_z = np.abs(rel_z_distances)

    # 3. Define your desired size range for markers
    max_size = 250  # Size for objects at z=0 (closest)
    min_size = 20   # Size for objects furthest away

    # 4. Normalize Z to range [0, 1]
    # Avoid division by zero if everything is on the plane
    max_z_val = np.max(abs_z)
    if max_z_val == 0:
        max_z_val = 1.0
    normalized_z = abs_z / max_z_val

    # 5. Inverse mapping: 0 (close) -> max_size, 1 (far) -> min_size
    marker_sizes = max_size - normalized_z * (max_size - min_size)
    return marker_sizes


def run_all_cluster_sims(interactions_file):
    df = pd.read_csv(interactions_file)

    stars_with_interactions = np.unique(df["particle1_id"])

    # values to create the initial disks
    N_disk = 2000
    N_disk=2000
    M_disk=0.01 | units.MSun
    R_min=1.0 | units.au
    R_max=100.0 | units.au
    q_out=-1.5

    t_sim = 1000 | units.yr

    last_interaction_time = 0 | units.yr
    last_star1_id = None
    last_star2_id = None

    for idx, mc_star in enumerate(stars_with_interactions):
        if idx == 4:
            break
        mask = (df["particle1_id"] == mc_star)
        interactions = df[mask]
        temp_star = None
        temp_disk = None
        for i in range(len(interactions)):
            print(f"Current Index i: {i}")
            print(f"Total rows in interactions: {len(interactions)}")
            S1id, S2id, M1, M2, R1, R2, T, REL_DIST, REL_VEL = get_initial_values_new(interactions.iloc[i])
            print(f"Interaction between stars {idx} and {i} (IDs: {S1id}; {S2id})")
            if (last_star1_id == S1id) and (last_star2_id == S2id) and (T - last_interaction_time < 1|units.Myr):
                continue
            if i == 0:
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
                star = temp_star
                disk = temp_disk

            perturber = Particles(1)
            perturber.mass = M2 
            perturber.radius = R2 
            perturber.position = REL_DIST   # Rest frame of other star
            perturber.velocity = REL_VEL 
            perturber.name = "PERTURBER"

            last_interaction_time = T
            last_star1_id = S1id
            last_star2_id = S2id

            save_name = f"{T.value_in(units.yr)}Myr_{idx}_{i}_{t_sim.value_in(units.yr)}_{M1.value_in(units.MSun):.3f}_{M2.value_in(units.MSun):.3f}"
            temp_star, temp_disk, pos_list, vel_list = simulate_disk_new(star, perturber, disk, f"DISK/DiskSave__{save_name}.amuse", t_sim=t_sim)
            filename_pos = f"DATA/FullRunNewPosAU__{save_name}.npy"
            filename_vel = f"DATA/FullRunNewVelKMS__{save_name}.npy"
            np.save(filename_pos, pos_list)
            np.save(filename_vel, vel_list)
            load_and_plot_data(filename_pos, filename_vel, PlotName=f"DEBUG_{idx}_{i}")



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
    run_all_cluster_sims("Interactions stopping conditions_100Myr.csv")
    fp1 = "Data//FullRunPosAU__99783.7073997Myr_1_0_500_9.331_2.330.npy"
    fv1 = "Data//FullRunVelKMS__99783.7073997Myr_1_0_500_9.331_2.330.npy"
    fp2 = "Data/FullRunPosAU__100266.667263Myr_1_1_500_9.331_2.330.npy"
    fv2 = "Data/FullRunVelKMS__100266.667263Myr_1_1_500_9.331_2.330.npy"
    fp3 = "Data/FullRunPosAU__100324.412464Myr_1_2_500_9.331_2.330.npy"
    fv3 = "Data/FullRunVelKMS__100324.412464Myr_1_2_500_9.331_2.330.npy"
    #load_and_animate_data(fp1, fv1)
    #load_and_animate_data(fp2, fv2)
    #load_and_animate_data(fp3,fv3)


