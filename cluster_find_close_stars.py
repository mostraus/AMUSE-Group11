import numpy 
from matplotlib import pyplot
from amuse.units import units, constants
from amuse.lab import Particles
from amuse.lab import nbody_system
from amuse.couple import bridge
from amuse.community.huayno import Huayno
from amuse.ic.salpeter import new_salpeter_mass_distribution
from amuse.ic.plummer import new_plummer_model
from amuse.ic.fractalcluster import new_fractal_cluster_model
from amuse.lab import Particle, Particles, units, nbody_system
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.animation as FuncAnimation
import csv
from collections import defaultdict, Counter

class MilkyWay_galaxy(object):
    def __init__(
        self, 
        Mb=1.40592e10 | units.MSun,
        Md=8.5608e10 | units.MSun,
        Mh=1.07068e11 | units.MSun,
    ):
        self.Mb = Mb
        self.Md = Md
        self.Mh = Mh

    def get_potential_at_point(self,eps,x,y,z):
        r = (x**2+y**2+z**2)**0.5
        R = (x**2+y**2)**0.5
        # bulge
        b1 = 0.3873 | units.kpc
        pot_bulge = -constants.G*self.Mb/(r**2+b1**2)**0.5 
        # disk
        a2 = 5.31 | units.kpc
        b2 = 0.25 | units.kpc
        pot_disk = -constants.G*self.Md/(R**2 + (a2+ (z**2+ b2**2)**0.5 )**2 )**0.5
        # halo
        a3 = 12.0 | units.kpc
        cut_off = 100 | units.kpc
        d1= r/a3
        c = 1+ (cut_off/a3)**1.02
        pot_halo = (
            -constants.G * (self.Mh/a3) * d1**1.02/(1+ d1**1.02)
            -(constants.G*self.Mh/(1.02*a3))
            * (
                -1.02 / c + numpy.log(c) + 1.02/(1+d1**1.02) 
                - numpy.log(1.0 + d1**1.02)
            )
        )
        return 2*(pot_bulge+pot_disk+ pot_halo)  # multiply by 2 because it is a rigid potential
    
    def get_gravity_at_point(self, eps, x,y,z): 
        r = (x**2+y**2+z**2)**0.5
        R = (x**2+y**2)**0.5
        # bulge
        b1 = 0.3873 | units.kpc
        force_bulge = -constants.G*self.Mb/(r**2+b1**2)**1.5 
        # disk
        a2 = 5.31 | units.kpc
        b2 = 0.25 | units.kpc
        d = a2+ (z**2+ b2**2)**0.5
        force_disk = -constants.G*self.Md/(R**2+ d**2 )**1.5
        # halo
        a3 = 12.0 | units.kpc
        d1 = r/a3
        force_halo = -constants.G*self.Mh*d1**0.02/(a3**2*(1+d1**1.02))
       
        ax = force_bulge*x + force_disk*x  + force_halo*x/r
        ay = force_bulge*y + force_disk*y  + force_halo*y/r
        az = force_bulge*z + force_disk*d*z/(z**2 + b2**2)**0.5 + force_halo*z/r 

        return ax,ay,az
    

import numpy as np
from amuse.units import units

def evolve_cluster_and_store(gravity, cluster, channel, times, number_of_stars, save_to_file=False):
    """
    Evolve the cluster over the given times, store positions and velocities of stars.

    Parameters
    ----------
    gravity : AMUSE gravity solver instance
    cluster : AMUSE particle set
    channel : AMUSE channel to copy data
    times : array of times to evolve
    number_of_stars : int
    save_to_file : bool, optional
        If True, saves the positions and velocities to .npz file.
    
    Returns
    -------
    cluster_over_time_x, cluster_over_time_y, cluster_over_time_z : np.ndarray
    cluster_over_time_vx, cluster_over_time_vy, cluster_over_time_vz : np.ndarray
    """

    # Initialize arrays to store positions and velocities
    cluster_over_time_x = np.zeros((number_of_stars, times.shape[0]))
    cluster_over_time_y = np.zeros((number_of_stars, times.shape[0]))
    cluster_over_time_z = np.zeros((number_of_stars, times.shape[0]))
    cluster_over_time_vx = np.zeros((number_of_stars, times.shape[0]))
    cluster_over_time_vy = np.zeros((number_of_stars, times.shape[0]))
    cluster_over_time_vz = np.zeros((number_of_stars, times.shape[0]))

    for j, time in enumerate(times):
        gravity.evolve_model(time)      
        print(f"Evolving time: {gravity.time}")
        channel.copy()    
        
        cluster_over_time_x[:, j] = cluster.x.value_in(units.parsec)
        cluster_over_time_y[:, j] = cluster.y.value_in(units.parsec)
        cluster_over_time_z[:, j] = cluster.z.value_in(units.parsec)
        cluster_over_time_vx[:, j] = cluster.vx.value_in(units.kms)
        cluster_over_time_vy[:, j] = cluster.vy.value_in(units.kms)
        cluster_over_time_vz[:, j] = cluster.vz.value_in(units.kms)

    print("Done evolving cluster, cleaning up...")
    gravity.stop()

    # Optionally save to file
    if save_to_file:
        np.savez("cluster_over_time.npz",
                 x=cluster_over_time_x,
                 y=cluster_over_time_y,
                 z=cluster_over_time_z,
                 vx=cluster_over_time_vx,
                 vy=cluster_over_time_vy,
                 vz=cluster_over_time_vz)
        print("Saved cluster positions and velocities to cluster_over_time.npz")

    return cluster_over_time_x, cluster_over_time_y, cluster_over_time_z, \
           cluster_over_time_vx, cluster_over_time_vy, cluster_over_time_vz


import matplotlib.pyplot as plt

def plot_cluster_trajectories(cluster_over_time_x, cluster_over_time_y, cluster_over_time_z, 
                              number_of_stars, convert_to_kpc=True, show_legend=False):
    """
    Plot the trajectories of stars in the cluster in the x-y plane.

    Parameters
    ----------
    cluster_over_time_x, cluster_over_time_y, cluster_over_time_z : np.ndarray
        Arrays of shape (number_of_stars, time_steps) with positions.
    number_of_stars : int
        Number of stars to plot.
    convert_to_kpc : bool, optional
        If True, convert parsecs to kiloparsecs for plotting (default True).
    show_legend : bool, optional
        If True, show legend with star indices (default False).
    """

    plt.figure(figsize=(8,6))

    for i in range(number_of_stars):
        x_vals = cluster_over_time_x[i, :]
        y_vals = cluster_over_time_y[i, :]

        if convert_to_kpc:
            x_vals = x_vals / 1000.0  # 1 kpc = 1000 pc
            y_vals = y_vals / 1000.0

        plt.plot(x_vals, y_vals, lw=1, alpha=0.7, label=f"Star {i}")

    plt.xlabel("x [kpc]" if convert_to_kpc else "x [pc]")
    plt.ylabel("y [kpc]" if convert_to_kpc else "y [pc]")
    plt.title("Trajectories of stars in cluster")
    
    if show_legend:
        plt.legend()
    
    plt.grid(True)
    plt.show()


def animate_cluster_trajectories(cluster_over_time_x, cluster_over_time_y, cluster_over_time_z, 
                                 number_of_stars, convert_to_kpc=True, interval=50, save_file=None):
    """
    Animate the motion of stars along their trajectories in the x-y plane.

    Parameters
    ----------
    cluster_over_time_x, cluster_over_time_y, cluster_over_time_z : np.ndarray
        Arrays of shape (number_of_stars, time_steps) with positions.
    number_of_stars : int
        Number of stars in the cluster.
    convert_to_kpc : bool, optional
        If True, convert parsecs to kiloparsecs for plotting (default True).
    interval : int, optional
        Delay between frames in milliseconds (default 50 ms).
    save_file : str or None, optional
        If given, save animation to this file (e.g., 'cluster.mp4').
    """
    # Convert units if needed
    if convert_to_kpc:
        x_data = cluster_over_time_x / 1000.0
        y_data = cluster_over_time_y / 1000.0
    else:
        x_data = cluster_over_time_x.copy()
        y_data = cluster_over_time_y.copy()

    time_steps = x_data.shape[1]

    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_xlim(x_data.min() * 1.1, x_data.max() * 1.1)
    ax.set_ylim(y_data.min() * 1.1, y_data.max() * 1.1)
    ax.set_xlabel("x [kpc]" if convert_to_kpc else "x [pc]")
    ax.set_ylabel("y [kpc]" if convert_to_kpc else "y [pc]")
    ax.set_title("Animated Trajectories of Stars in Cluster")
    ax.grid(True)

    # Create a plot line for each star
    lines = [ax.plot([], [], 'o', markersize=4, alpha=0.8)[0] for _ in range(number_of_stars)]

    # Animation function
    def update(frame):
        for i, line in enumerate(lines):
            line.set_data(x_data[i, frame], y_data[i, frame])
        return lines

    anim = FuncAnimation(fig, update, frames=time_steps, interval=interval, blit=True)

    if save_file:
        anim.save(save_file, writer='ffmpeg', fps=30)
        print(f"Animation saved to {save_file}")
    else:
        plt.show()


def reactions(cluster, number_of_stars, R_hill ,times, cluster_over_time_x, cluster_over_time_y, cluster_over_time_z,
              cluster_over_time_vx, cluster_over_time_vy, cluster_over_time_vz):
    
    reactions = numpy.zeros((number_of_stars,number_of_stars))
    for main_star in range(number_of_stars):
        for j in range(number_of_stars):
            a = R_hill * (3*cluster[j].mass/cluster[main_star].mass)**(1/3)
            a = 1000 | units.au
            reactions[main_star, j] = a.value_in(units.parsec)
    #print(reactions[9][8])

    # Optimize interaction search: iterate over time steps, using KD-tree
    max_radius = reactions.max()  # scalar (parsec)
    T = times.shape[0]
    N = number_of_stars

    # Prepare positions per time: shape (T, N, 3)
    positions_by_time = numpy.stack((cluster_over_time_x.T,
                                    cluster_over_time_y.T,
                                    cluster_over_time_z.T), axis=2)
    velocity_by_time = numpy.stack((cluster_over_time_vx.T,
                                    cluster_over_time_vy.T,
                                    cluster_over_time_vz.T), axis=2)

    # Initialize list of interactions for each star
    interactions_list = [[] for _ in range(N)]

    for ti in range(T):
        pos = positions_by_time[ti]  # (N,3), units: parsec
        vel = velocity_by_time[ti]   # (N,3), units: km/s
        
        # Build KD-tree for fast neighbor search
        tree = cKDTree(pos)
        
        # neighbors_list[i] contains indices of stars within max_radius of star i (includes i itself)
        neighbors_list = tree.query_ball_tree(tree, r=float(max_radius))
        
        for i, neigh in enumerate(neighbors_list):
            if len(neigh) <= 1:
                continue
            
            # Remove self from neighbors
            neigh = [j for j in neigh if j != i]
            if not neigh:
                continue
            
            neigh = numpy.array(neigh, dtype=int)
            
            # Differences in positions and velocities
            dx = pos[neigh, 0] - pos[i, 0]
            dy = pos[neigh, 1] - pos[i, 1]
            dz = pos[neigh, 2] - pos[i, 2]
            dvx = vel[neigh, 0] - vel[i, 0]
            dvy = vel[neigh, 1] - vel[i, 1]
            dvz = vel[neigh, 2] - vel[i, 2]
            
            # Distances and relative velocities
            dists = numpy.hypot(dx, dy, dz)   # distances in parsec
            vels = numpy.hypot(dvx, dvy, dvz) # relative velocities in km/s
            
            # Filter by per-pair threshold
            mask = dists < reactions[i, neigh]
            hit_idx = neigh[mask]
            hit_dists = dists[mask]
            hit_vels = vels[mask]
            
            # Save interactions
            for j_idx, dist, relativ_vel in zip(hit_idx, hit_dists, hit_vels):
                interactions_list[i].append((int(i), int(j_idx), int(ti), float(dist), float(relativ_vel)))
        
        if ti % 50 == 0:
            print(f"time {ti}/{T}")
    print("done, stars with interactions:", sum(1 for lst in interactions_list if lst))
    interactions_list_filtered = [lst for lst in interactions_list if lst]
    print(interactions_list_filtered)

    return interactions_list_filtered

def save(interactions_list_filtered):
    all_interactions = [item for sublist in interactions_list_filtered for item in sublist]

    with open("interactions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["star_i", "star_j", "time_index", "distance_pc", "rel_velocity_kms"])
        # Write data
        writer.writerows(all_interactions)

    print("Saved interactions to interactions.csv")

## SETTING THE INITIAL CONDITIONS FOR THE CLUSTER IN THE GALAXY

#setting the initial conditions - number of stars, velocity of the cluster, size of a cluster, position of a cluster
number_of_stars = 100
vel_cluster = 40

masses = new_salpeter_mass_distribution(number_of_stars, 1|units.MSun, 30|units.MSun)
#set to a 1000 particles cluster 
converter = nbody_system.nbody_to_si(masses.sum(), 1|units.parsec)

cluster = new_fractal_cluster_model(number_of_stars, converter)
cluster.mass = masses
cluster.move_to_center()
cluster.position += [8.5, 0, 0] | units.kpc
cluster.velocity += [0, vel_cluster, 0] |units.kms
#Setting hill radius - initial condition

R_hill = 1000 | units.au
R_hill_pc = R_hill.value_in(units.parsec)

#print(R_hill_pc)

# print(cluster.x.value_in(units.parsec).min())
# print(cluster.x.value_in(units.parsec).max())

# print(cluster[2].mass)


#evolving the system in the Milky Way potential
gravity_code = Huayno(converter)
gravity_code.particles.add_particles(cluster)
channel = gravity_code.particles.new_channel_to(cluster)
MWG = MilkyWay_galaxy()

gravity = bridge.Bridge(use_threading=False)
gravity.add_system(gravity_code, (MWG,) )
gravity.timestep = 1|units.Myr

times = numpy.arange(0.0, 100, 1) | units.Myr


#usage of the functions 
cluster_over_time_x, cluster_over_time_y, cluster_over_time_z, cluster_over_time_vx, cluster_over_time_vy, cluster_over_time_vz = evolve_cluster_and_store(
    gravity, cluster, channel, times, number_of_stars, save_to_file=True
)

interactions_list_filtered = reactions(cluster, number_of_stars, R_hill ,times, cluster_over_time_x, cluster_over_time_y, cluster_over_time_z,
              cluster_over_time_vx, cluster_over_time_vy, cluster_over_time_vz)



save(interactions_list_filtered)



# Flatten all interaction entries and sort the indices in each pair
all_interactions = [tuple(sorted((i, j))) 
                    for sublist in interactions_list 
                    for (i, j, _, _, _) in sublist]

# 1️⃣ Count how many times each pair of stars interacted
pair_counts = Counter(all_interactions)

print(f"Total unique interacting pairs: {len(pair_counts)}")
print("Top 10 most frequent pairs:")
for pair, count in pair_counts.most_common(10):
    print(f"Stars {pair[0]}–{pair[1]}: {count} interactions")

# 2️⃣ Count unique interactions for each star
# Using a dictionary of sets: key = star index, value = set of stars it interacted with
unique_interactions = defaultdict(set)
for i, j in pair_counts.keys():  # keys are unique pairs
    unique_interactions[i].add(j)
    unique_interactions[j].add(i)  # interactions are bidirectional

# Compute the number of unique interactions for each star
unique_counts = {star: len(neighs) for star, neighs in unique_interactions.items()}

# Find the star with the largest number of unique interactions
max_star = max(unique_counts, key=lambda k: unique_counts[k])
max_count = unique_counts[max_star]

print(f"\nStar {max_star} has the largest number of unique interactions: {max_count}")

# Top 5 stars with the largest number of unique interactions
top5 = sorted(unique_counts.items(), key=lambda x: x[1], reverse=True)[:5]
print("\nTop 5 stars with the largest number of unique interactions:")
for star, count in top5:
    print(f"Star {star}: {count} unique interactions")


