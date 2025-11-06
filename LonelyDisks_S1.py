from amuse.units import units, constants
from amuse.lab import *
from amuse.couple import bridge

import numpy as np
from matplotlib import pyplot as plt
import csv

from galaxy_potentials import MilkyWay_galaxy
from cluster_creator import make_cluster_plummer


from amuse.lab import constants

def get_star_hill_radius_in_galaxy(star, galaxy_potential):
    """
    Calculates the Hill radius of a star particle relative to the 
    host galaxy potential it is orbiting.
    
    Args:
        star: The AMUSE particle (with star.mass and star.position).
        galaxy_potential: An instance of your MilkyWay_galaxy class.
    
    Returns:
        The Hill radius as an AMUSE quantity (e.g., in units.pc).
    """
    
    # 1. Get star's mass and position
    m_star = star.mass
    print(m_star.in_(units.MSun))
    x, y, z = star.position
    
    # 2. Get distance from galactic center (assuming center is at 0,0,0)
    r_from_center = star.position.length()
    
    if r_from_center == 0 | units.kpc:
        # Cannot calculate at the exact center, return zero
        return 0 | units.AU

    # 3. Get total gravity (acceleration) from the galaxy at that point
    #    Your get_gravity_at_point needs 'eps' as the first arg.
    ax, ay, az = galaxy_potential.get_gravity_at_point(0, x, y, z)
    accel_mag = (ax**2 + ay**2 + az**2).sqrt()

    # 4. Calculate the enclosed mass of the galaxy at that distance
    #    From F = G*M_enc*m / r^2 and F = m*a  =>  a = G*M_enc / r^2
    #    Therefore, M_enc = a * r^2 / G
    M_galaxy_enclosed = (accel_mag * r_from_center**2) / constants.G
    print(M_galaxy_enclosed.in_(units.MSun))
    
    if M_galaxy_enclosed <= 0 | units.MSun:
        # Avoid division by zero or negative mass
        return 0 | units.AU

    # 5. Calculate the Hill Radius
    #    The 'cbrt()' function is 'cube root'
    hill_radius = np.cbrt((r_from_center * (m_star / (3.0 * M_galaxy_enclosed))).value_in(units.AU))
    print(hill_radius)
    return hill_radius | units.AU


def simulate_cluster_stopping_condition(galaxy_potential, Nstars, R_ini, pos_ini, vel_ini, times, enc_dist, output_filename="cluster_encounters.csv"):

    cluster = make_cluster_plummer(N_stars=Nstars, plummer_radius=R_ini)
    cluster.position += pos_ini
    cluster.velocity += vel_ini
    cluster.radius = enc_dist / 2.0

    converter_S1 = nbody_system.nbody_to_si(cluster.mass.sum(), cluster.position.length())
    gravity_code_cluster = ph4(converter_S1)
    gravity_code_cluster.parameters.epsilon_squared = eps_soft*eps_soft
    gravity_code_cluster.particles.add_particles(cluster)
    ch2_cluster = gravity_code_cluster.particles.new_channel_to(cluster)

    gravity_code_cluster.stopping_conditions.collision_detection.enable()

    gravity_cluster = bridge.Bridge(use_threading=False)
    gravity_cluster.add_system(gravity_code_cluster, (galaxy_potential,))
    gravity_cluster.timestep = 1 | units.Myr

    # Use a dictionary to store paths: {star_index: [[x_list], [y_list]]}
    trajectories = {i: ([], []) for i in range(Nstars)}
    # Use an array to save the cluster history
    cluster_history = np.zeros_like(times)

    FILE_HEADER = ["time_myr", "star1_id", "star2_id", "mass1_msun", "mass2_msun", "min_distance_au", "rel_velocity_kms", 
                   "x1_pc", "y1_pc", "z1_pc", "vx1_kms", "vy1_kms", "vz1_kms", 
                   "x2_pc", "y2_pc", "z2_pc", "vx2_kms", "vy2_kms", "vz2_kms",
                   ]
    print(f"Logging Encounters to: {output_filename}")

    with open(output_filename, "w", newline="") as csvfile:     # open file before loop
        writer = csv.DictWriter(csvfile, fieldnames=FILE_HEADER)
        writer.writeheader()

        for i,target_time in enumerate(times):
            while gravity_cluster.model_time < target_time:     # smaller loop to get all encounters
                gravity_cluster.evolve_model(target_time)
                if gravity_code_cluster.stopping_conditions.collision_detection.is_set():   # Check if encounter occured
                    time_now = gravity_cluster.model_time
                    colliding_particles = gravity_code_cluster.stopping_conditions.collision_detection.particles()
                    print(colliding_particles)
                    if len(colliding_particles) == 2:
                        star1 = colliding_particles[0]   # get stars properties
                        star2 = colliding_particles[1]
                        p1 = gravity_code_cluster.particles[star1.key]
                        p2 = gravity_code_cluster.particles[star2.key]
                        #p1 = star1.key_in_set.as_particle_in_set(gravity_code_cluster.particles)
                        #p2 = star2.key_in_set.as_particle_in_set(gravity_code_cluster.particles)

                        pos_rel = p1.position - p2.position     # calculate relative properties at closest approach
                        vel_rel = p1.velocity - p2.velocity
                        dist_min = pos_rel.length()
                        vel_rel_val = vel_rel.length()

                        encounter_data = {
                            "time_myr": time_now.value_in(units.Myr), 
                            "star1_id": p1.key, "star2_id": p2.key, 
                            "mass1_msun": p1.mass.value_in(units.MSun), "mass2_msun": p2.mass.value_in(units.MSun), 
                            "min_distance_au": dist_min.value_in(units.AU), "rel_velocity_kms": vel_rel_val.value_in(units.kms), 
                            "x1_pc": p1.x.value_in(units.pc), "y1_pc": p1.y.value_in(units.pc), "z1_pc": p1.z.value_in(units.pc), 
                            "vx1_kms": p1.vx.value_in(units.kms), "vy1_kms": p1.vy.value_in(units.kms), "vz1_kms": p1.vz.value_in(units.kms), 
                            "x2_pc": p2.x.value_in(units.pc), "y2_pc": p2.y.value_in(units.pc), "z2_pc": p2.z.value_in(units.pc), 
                            "vx2_kms": p2.vx.value_in(units.kms), "vy2_kms": p2.vy.value_in(units.kms), "vz2_kms": p2.vz.value_in(units.kms)
                        }
                        writer.writerow(encounter_data)
                        print(f"NICE: Encounter at {gravity_cluster.model_time.in_(units.Myr)} with {len(colliding_particles)} particles, data preview: {encounter_data["time_myr"]}, {encounter_data["star1_id"]}, {encounter_data["star2_id"]}, {encounter_data["min_distance_au"]}!")
                    else:
                        print(f"WARNING: Encounter at {gravity_cluster.model_time.in_(units.Myr)} with {len(colliding_particles)} particles, skipping!")

            ch2_cluster.copy()
            cluster_history[i] = cluster.copy()
            for j in range(Nstars):
                trajectories[j][0].append(cluster[j].x.value_in(units.kpc))
                trajectories[j][1].append(cluster[j].y.value_in(units.kpc))

    gravity_code_cluster.stopping_conditions.collision_detection.disable()
    print("Encounter logging complete.")
    
    return trajectories, cluster_history


def simulate_cluster(galaxy_potential, Nstars, R_ini, pos_ini, vel_ini, times, enc_dist, output_filename="cluster_encounters.csv"):

    cluster = make_cluster_plummer(N_stars=Nstars, plummer_radius=R_ini)
    #enc_dist = 10 * np.max([get_star_hill_radius_in_galaxy(star, galaxy_potential) for star in cluster])
    cluster.position += pos_ini
    cluster.velocity += vel_ini

    converter_S1 = nbody_system.nbody_to_si(cluster.mass.sum(), cluster.position.length())
    gravity_code_cluster = Hermite(converter_S1)
    gravity_code_cluster.particles.add_particles(cluster)
    ch2_cluster = gravity_code_cluster.particles.new_channel_to(cluster)

    gravity_cluster = bridge.Bridge(use_threading=False)
    gravity_cluster.add_system(gravity_code_cluster, (galaxy_potential,))
    gravity_cluster.timestep = 1 | units.Myr

    # Use a dictionary to store paths: {star_index: [[x_list], [y_list]]}
    trajectories = {i: ([], []) for i in range(Nstars)}
    # Use an array to save the cluster history
    cluster_history = np.zeros_like(times)

    FILE_HEADER = ["time_myr", "star1_id", "star2_id", "mass1_msun", "mass2_msun", "min_distance_au", "rel_velocity_kms", 
                   "x1_pc", "y1_pc", "z1_pc", "vx1_kms", "vy1_kms", "vz1_kms", 
                   "x2_pc", "y2_pc", "z2_pc", "vx2_kms", "vy2_kms", "vz2_kms",
                   ]
    print(f"Logging Encounters to: {output_filename}")

    with open(output_filename, "w", newline="") as csvfile:     # open file before loop
        writer = csv.DictWriter(csvfile, fieldnames=FILE_HEADER)
        writer.writeheader()

        for i,target_time in enumerate(times):
            while gravity_cluster.model_time < target_time:     # smaller loop to get all encounters
                gravity_cluster.evolve_model(target_time)
                time_now = gravity_cluster.model_time

                distances = np.zeros((Nstars,Nstars))     # matrix with the distances to the stars element d01 is distance from star 1 to star 2, symmetric matrix with 0 on diagonal
                current_particles = gravity_code_cluster.particles
                for k in range(1,Nstars):
                    for l in range(k):
                        x_sq = (current_particles[k].x.value_in(units.AU) - current_particles[l].x.value_in(units.AU))**2
                        y_sq = (current_particles[k].y.value_in(units.AU) - current_particles[l].y.value_in(units.AU))**2
                        z_sq = (current_particles[k].z.value_in(units.AU) - current_particles[l].z.value_in(units.AU))**2
                        d = np.sqrt(x_sq + y_sq + z_sq)
                        distances[k][l] = d

                for m,d in enumerate(distances.flatten()):
                    if d > 0 and d < enc_dist.value_in(units.AU):
                        a = int(m / Nstars)
                        b = m % Nstars
                        print(f"Close encounter between stars {a+1} and {b+1} at a distance of {d}AU!")
                        p1 = current_particles[a]
                        p2 = current_particles[b]
     
                        vel_rel = p1.velocity - p2.velocity     # calculate relative properties at closest approach
                        vel_rel_val = vel_rel.length()

                        encounter_data = {
                            "time_myr": time_now.value_in(units.Myr), 
                            "star1_id": p1.key, "star2_id": p2.key, 
                            "mass1_msun": p1.mass.value_in(units.MSun), "mass2_msun": p2.mass.value_in(units.MSun), 
                            "min_distance_au": d, "rel_velocity_kms": vel_rel_val.value_in(units.kms), 
                            "x1_pc": p1.x.value_in(units.pc), "y1_pc": p1.y.value_in(units.pc), "z1_pc": p1.z.value_in(units.pc), 
                            "vx1_kms": p1.vx.value_in(units.kms), "vy1_kms": p1.vy.value_in(units.kms), "vz1_kms": p1.vz.value_in(units.kms), 
                            "x2_pc": p2.x.value_in(units.pc), "y2_pc": p2.y.value_in(units.pc), "z2_pc": p2.z.value_in(units.pc), 
                            "vx2_kms": p2.vx.value_in(units.kms), "vy2_kms": p2.vy.value_in(units.kms), "vz2_kms": p2.vz.value_in(units.kms)
                        }
                        writer.writerow(encounter_data)
                    
            ch2_cluster.copy()
            cluster_history[i] = cluster.copy()
            for j in range(Nstars):
                trajectories[j][0].append(cluster[j].x.value_in(units.kpc))
                trajectories[j][1].append(cluster[j].y.value_in(units.kpc))

    print("Encounter logging complete.")
    return trajectories, cluster_history


def plot_cluster_in_galaxy(trajectories, pos_ini, Nstars, ax):
    for i in range(Nstars):
        ax.plot(trajectories[i][0], trajectories[i][1])
    pos_ini = pos_ini.value_in(units.kpc)
    ax.scatter([pos_ini[0]], [pos_ini[1]], color="red", s=10, label="initial position of cluster")
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    ax.legend()
    ax.set_title("Cluster Motion")


def plot_cluster_evolution(trajectories, times, cluster_history, Nstars, ax):
    COMs = [cluster.center_of_mass() for cluster in cluster_history]
    for i in range(Nstars):
        traj_x = np.array(trajectories[i][0]) - np.array([COM.x.value_in(units.kpc) for COM in COMs])
        traj_y = np.array(trajectories[i][1]) - np.array([COM.y.value_in(units.kpc) for COM in COMs])
        ax.scatter(traj_x, times.value_in(units.Myr), traj_y)
    ax.set_xlabel('X [kpc]')
    ax.set_ylabel('Time [Myr]')
    ax.set_zlabel('Y [kpc]')
    ax.set_title("Cluster Evolution")




def main():
    MWG = MilkyWay_galaxy()
    NSTARS = 10
    INITIAL_RADIUS = 10     # in pc

    INITIAL_POS = [8.5, 0, 0] | units.kpc
    INITIAL_VEL = [0, 120, 0] | units.kms

    time_end = 1000   # Myr
    timestep = 1    # Myr
    TIMES = np.arange(0., time_end, timestep) | units.Myr

    ENCOUNTER_DISTANCE = 2 | units.pc

    trajectories, cluster_history = simulate_cluster(MWG, NSTARS, INITIAL_RADIUS, INITIAL_POS, INITIAL_VEL, TIMES, ENCOUNTER_DISTANCE)
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2,projection='3d')
    plot_cluster_in_galaxy(trajectories, INITIAL_POS, NSTARS, ax1)
    plot_cluster_evolution(trajectories, TIMES, cluster_history, NSTARS, ax2)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
