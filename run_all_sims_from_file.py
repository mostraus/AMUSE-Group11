import pandas as pd
import numpy as np
from amuse.lab import *
from amuse.ext.protodisk import ProtoPlanetaryDisk
from HydroExample2 import simulate_hydro_disk, load_and_plot_data, load_and_animate_data, get_initial_values_new, simulate_disk_new, simulate_2disk_new
from functions import get_bound_particles_fraction, bound_fraction_plot


def run_all_cluster_sims(interactions_file):
    df = pd.read_csv(interactions_file)

    stars_with_interactions = np.unique(df["particle1_id"])

    # values to create the initial disks
    N_disk = 2000
    M_disk=0.01 | units.MSun
    R_min=1.0 | units.au
    R_max=100.0 | units.au
    q_out=-1.5

    t_sim = 1000 | units.yr

    # memory to skip when the same interaction was logged multiple times
    last_interaction_time = 0 | units.yr
    last_star1_id = None
    last_star2_id = None

    # lists for the distance vs particles lost plot
    enc_dists = []
    frac_lost = []

    for idx, mc_star in enumerate(stars_with_interactions):
        #if idx == 4:
        #    break
        mask = (df["particle1_id"] == mc_star)
        interactions = df[mask]
        temp_star = None
        temp_disk = None
        for i in range(len(interactions)):
            #print(f"Current Index i: {i}")
            #print(f"Total rows in interactions: {len(interactions)}")
            S1id, S2id, M1, M2, R1, R2, T, REL_DIST, REL_VEL = get_initial_values_new(interactions.iloc[i])
            print(f"Interaction between stars {idx} and {i} (IDs: {S1id}; {S2id})")
            if (last_star1_id == S1id) and (last_star2_id == S2id) and (T - last_interaction_time < 1|units.Myr):
                print(f"skip interaction {i} between {S1id} and {S2id} because only {T - last_interaction_time} have passed")
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

            # find closest approach and fraction of particles lost during this encounter
            min_dist = np.min(np.absolute(np.linalg.norm(pos_list[1], axis=1) - np.linalg.norm(pos_list[0], axis=1)))
            enc_dists.append(min_dist)
            bound_frac_end = get_bound_particles_fraction(M1, pos_list[0,-1,:], vel_list[0,-1,:], pos_list[2:,-1,:], vel_list[2:,-1,:])
            bound_frac_start = get_bound_particles_fraction(M1, pos_list[0,-1,:], vel_list[0,-1,:], pos_list[2:,0,:], vel_list[2:,0,:])
            frac_lost.append(bound_frac_end / bound_frac_start)

    bound_fraction_plot(enc_dists, frac_lost, PlotName=f"DistVSLost__{save_name}")


def run_sim_2disk(interactions_file, s1id, s2id, index=0, t_sim=500|units.yr):
    df = pd.read_csv(interactions_file)
    mask = ((df["particle1_id"] == s1id) & (df["particle2_id"] == s2id))
    df_filtered = df[mask]
    arr = df_filtered.iloc[index]
    S1id, S2id, M1, M2, R1, R2, T, REL_DIST, REL_VEL = get_initial_values_new(arr)

    # create Stars and Disks
    star1 = Particles(1)
    star1.mass = M1
    star1.radius = R1 
    star1.position = (0, 0, 0) | units.au    # Rest frame of this star
    star1.velocity = (0, 0, 0) | units.kms
    star1.name = "STAR"

    star2 = Particles(1)
    star2.mass = M2 
    star2.radius = R2 
    star2.position = REL_DIST   # Rest frame of other star
    star2.velocity = REL_VEL 
    star2.name = "PERTURBER"

    # define disk values
    N_disk = 2000
    M_disk=0.01 | units.MSun
    R_min=1.0 | units.au
    R_max=100.0 | units.au
    q_out=-1.5

    hydro_converter1 = nbody_system.nbody_to_si(M1, R_max)
    hydro_converter2 = nbody_system.nbody_to_si(M2, R_max)

    disk1 = ProtoPlanetaryDisk(N_disk, 
                    convert_nbody=hydro_converter1, 
                    Rmin=R_min/R_max, 
                    radius_min= R_min/R_max,
                    Rmax=1, 
                    radius_max= 1,
                    q_out=q_out, 
                    discfraction=M_disk/M1).result
    
    disk2 = ProtoPlanetaryDisk(N_disk, 
                    convert_nbody=hydro_converter2, 
                    Rmin=R_min/R_max, 
                    radius_min= R_min/R_max,
                    Rmax=1, 
                    radius_max= 1,
                    q_out=q_out, 
                    discfraction=M_disk/M1).result
    
    disk2.position += REL_DIST
    disk2.velocity += REL_VEL

    # make simulation
    end_star, end_disk1, end_disk2, pos_list, vel_list = simulate_2disk_new(star1, star2, disk1, disk2, "", t_sim)

    save_name = f"{T.value_in(units.yr)}Myr_{S1id}_{S2id}_{t_sim.value_in(units.yr)}_{M1.value_in(units.MSun):.3f}_{M2.value_in(units.MSun):.3f}"
    filename_pos = f"DATA/2DiskRunPosAU__{save_name}.npy"
    filename_vel = f"DATA/2DiskRunVelKMS__{save_name}.npy"
    np.save(filename_pos, pos_list)
    np.save(filename_vel, vel_list)
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
    run_sim_2disk("Interactions stopping conditions_100Myr.csv", 113871276108901526, 15793569915568245741, t_sim=100|units.yr)
    #fp1 = "Data//FullRunPosAU__99783.7073997Myr_1_0_500_9.331_2.330.npy"
    #fv1 = "Data//FullRunVelKMS__99783.7073997Myr_1_0_500_9.331_2.330.npy"
    #fp2 = "Data/FullRunPosAU__100266.667263Myr_1_1_500_9.331_2.330.npy"
    #fv2 = "Data/FullRunVelKMS__100266.667263Myr_1_1_500_9.331_2.330.npy"
    #fp3 = "Data/FullRunPosAU__100324.412464Myr_1_2_500_9.331_2.330.npy"
    #fv3 = "Data/FullRunVelKMS__100324.412464Myr_1_2_500_9.331_2.330.npy"
    #load_and_animate_data(fp1, fv1)
    #load_and_animate_data(fp2, fv2)
    #load_and_animate_data(fp3,fv3)
    #fp = "Data/2DiskRunPosAU__966420.184921Myr_1.1387127610890152e+17_1.5793569915568245e+19_1000_9.331_1.687.npy"
    #fv = "Data/2DiskRunVelKMS__966420.184921Myr_1.1387127610890152e+17_1.5793569915568245e+19_1000_9.331_1.687.npy"
    #load_and_animate_data(fp, fv)


