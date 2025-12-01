import pandas as pd
import numpy as np
from amuse.lab import units
from HydroExample2 import simulate_hydro_disk, load_disk, load_and_plot_data


def run_all_sims_1disk(interactions_file):
    df = pd.read_csv(interactions_file)

    main_character_stars = df["star_i"]
    partner_stars = df["star_j"]
    used_mc_stars = []
    t_sim = 500.0 | units.yr

    print(main_character_stars)

    for i, mc in enumerate(main_character_stars):
        if i > 5:
            break
        p = partner_stars[i]
        print(f"RUNNING SIMULATION FOR {mc} AND {p}")
        if mc not in used_mc_stars:
            POSITIONS_LIST, VELOCITIES_LIST, info_array = simulate_hydro_disk(interactions_file, mc, p, t_sim=t_sim)
            used_mc_stars.append(mc)
        else:   # to see how the disk of the other star is affected we can just do all simulations
        #elif mc in used_mc_stars and mc < p:
            index = np.sum(used_mc_stars == mc)
            info = make_info_arr(df, mc, p, index)
            disk = load_disk(f"DISK/DiskSave_{info[0].value_in(units.Myr)}Myr_{float(mc)}_{float(p)}_{t_sim.value_in(units.yr)}yr_{info[1].value_in(units.MSun)}MSun_{info[2].value_in(units.MSun)}MSun.amuse")
            POSITIONS_LIST, VELOCITIES_LIST, info_array = simulate_hydro_disk(interactions_file, mc, p, give_Disk=disk, index=index, t_sim=t_sim)
            used_mc_stars.append(mc)
        #else:   # mc in used stars and mc > p --> this combination has already been simulated
        #    print(f"This combination ({mc} and {p}) has already been simulated as {p} and {mc}")
        
        ### Save the data ###
        filename_pos = f"Data/DiskDataPosAU_{info_array[0]}Myr_{info_array[1]}_{info_array[2]}_{info_array[3]}yr_{info_array[4]}MSun_{info_array[5]}MSun.npy"
        filename_vel = f"Data/DiskDataVelKMS_{info_array[0]}Myr_{info_array[1]}_{info_array[2]}_{info_array[3]}yr_{info_array[4]}MSun_{info_array[5]}MSun.npy"
        np.save(filename_pos, POSITIONS_LIST)
        np.save(filename_vel, VELOCITIES_LIST)
        load_and_plot_data(filename_pos, filename_vel, PlotName="FullRun")




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
    run_all_sims_1disk("interactions.csv")
