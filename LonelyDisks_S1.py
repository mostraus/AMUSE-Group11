from amuse.units import units, constants
from amuse.lab import *
from amuse.couple import bridge

import numpy as np
from matplotlib import pyplot as plt

from galaxy_potentials import MilkyWay_galaxy
from cluster_creator import make_cluster_plummer


def simulate_cluster(galaxy_potential, Nstars, R_ini, pos_ini, vel_ini, timesL):

    cluster = make_cluster_plummer(N_stars=Nstars, plummer_radius=R_ini)
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

    for i,time in enumerate(times):
        gravity_cluster.evolve_model(time)
        ch2_cluster.copy()
        cluster_history[i] = cluster.copy()
        for i in range(Nstars):
            trajectories[i][0].append(cluster[i].x.value_in(units.kpc))
            trajectories[i][1].append(cluster[i].y.value_in(units.kpc))
    
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
    INITIAL_VEL = [0, 10, 0] | units.kms

    time_end = 100   # Myr
    timestep = 1    # Myr
    TIMES = np.arange(0., time_end, timestep) | units.Myr

    trajectories, cluster_history = simulate_cluster(MWG, NSTARS, INITIAL_RADIUS, INITIAL_POS, INITIAL_VEL, TIMES)
    fig, ax = plt.subplots(figsize=(12,6))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2,projection='3d')
    plot_cluster_in_galaxy(trajectories, INITIAL_POS, NSTARS, ax1)
    plot_cluster_evolution(trajectories, TIMES, cluster_history, NSTARS, ax2)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
