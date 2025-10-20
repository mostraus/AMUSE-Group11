from amuse.lab import new_salpeter_mass_distribution, new_plummer_model
from amuse.units import units, constants
from amuse.lab import nbody_system
from amuse.couple import bridge
from amuse.community.huayno import Huayno
import numpy
from matplotlib import pyplot


def make_cluster_plummer(N_stars, plummer_radius):
    # 1. Define cluster parameters
    plummer_radius = plummer_radius | units.pc
    masses = new_salpeter_mass_distribution(N_stars, mass_min=1|units.MSun, mass_max=100|units.MSun)

    # 2. Generate the cluster
    # A converter is needed to scale the Plummer model to physical units
    converter_plummer = nbody_system.nbody_to_si(masses.sum(), plummer_radius)
    cluster = new_plummer_model(N_stars, convert_nbody=converter_plummer)
    cluster.mass = masses
    return cluster


def setup(galaxy, cluster, vel=[0,-210,0] | units.kms, N_stars=10, pos=[8.5, 0, 0] | units.kpc):    
    # 3. Set the cluster's initial orbital position and velocity
    # We move the entire cluster to the starting point of the single star
    cluster_pos = pos 
    cluster_vel = vel 
    cluster.position += cluster_pos
    cluster.velocity += cluster_vel

    # 4. Set up the AMUSE simulation
    # This converter is for the main N-body simulation
    converter_sim = nbody_system.nbody_to_si(cluster.mass.sum(), cluster_pos.length())
    gravity_code_cluster = Huayno(converter_sim)
    gravity_code_cluster.particles.add_particles(cluster)

    # Create a channel to copy data from the code back to our script
    ch_gc2l = gravity_code_cluster.particles.new_channel_to(cluster)

    # Set up the bridge to couple the cluster's gravity with the galaxy's potential
    gravity_cluster = bridge.Bridge(use_threading=False)
    gravity_cluster.add_system(gravity_code_cluster, (galaxy,))
    gravity_cluster.timestep = 1 | units.Myr

    return gravity_cluster, ch_gc2l, N_stars, cluster


def run(gravity_cluster, ch_gc2l, N_stars, cluster, length=250, timestep=1):
    # 5. Run the evolution loop
    times = numpy.arange(0., length, timestep) | units.Myr

    # Use a dictionary to store paths: {star_index: [[x_list], [y_list]]}
    trajectories = {i: ([], []) for i in range(N_stars)}

    for time in times:
        gravity_cluster.evolve_model(time)
        ch_gc2l.copy()  # Update particle info in the script
        
        # Loop through each star and record its position
        for i in range(N_stars):
            trajectories[i][0].append(cluster[i].x.value_in(units.kpc))
            trajectories[i][1].append(cluster[i].y.value_in(units.kpc))

    print("Simulation finished, cleaning up.")
    gravity_cluster.stop()
    return trajectories


def plot(trajectories, N_stars, cluster_pos):
    # 6. Plot the individual orbits
    pyplot.figure(figsize=(10, 10))

    # Plot each star's trajectory
    for i in range(N_stars):
        x_vals = trajectories[i][0]
        y_vals = trajectories[i][1]
        # Plot the first star with a label, others without to avoid clutter
        if i == 0:
            pyplot.plot(x_vals, y_vals, lw=1, alpha=0.8, label="Individual Star Orbits")
        else:
            pyplot.plot(x_vals, y_vals, lw=1, alpha=0.8)

    pyplot.scatter(cluster_pos[0].value_in(units.kpc), cluster_pos[1].value_in(units.kpc), color='red', s=50, zorder=5, label="Cluster Start Position")
    pyplot.title("Individual Star Orbits in a Cluster")
    pyplot.xlabel("x [kpc]")
    pyplot.ylabel("y [kpc]")
    pyplot.gca().set_aspect('equal', adjustable='box')
    pyplot.grid(True)
    pyplot.legend()
    pyplot.show()
