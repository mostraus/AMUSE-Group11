import numpy as np
from matplotlib import pyplot as plt

from amuse.lab import *
from amuse.units import nbody_system
from amuse.ext.bridge import bridge

from galaxy_potentials import MilkyWay_galaxy

def evolve_cluster_in_galaxy(
    number_of_stars,
    cluster_mass,
    virial_radius,
    end_time,
    time_step,
    initial_position,
    initial_velocity,
    galaxy_potential="MWPotential"
):
    """
    Evolves a star cluster in an external galactic potential using AMUSE Bridge.

    Args:
        number_of_stars (int): Number of stars in the cluster.
        cluster_mass (Quantity): Total mass of the star cluster (e.g., 1000 | units.MSun).
        virial_radius (Quantity): Virial radius of the cluster (e.g., 1.0 | units.parsec).
        end_time (Quantity): The total simulation time (e.g., 100 | units.Myr).
        time_step (Quantity): The time step for the main simulation loop (e.g., 1 | units.Myr).
        initial_position (Quantity tuple): Initial position of the cluster's CoM, e.g., [10, 0, 0] | units.kpc.
        initial_velocity (Quantity tuple): Initial velocity of the cluster's CoM, e.g., [0, 220, 0] | units.kms.
        galaxy_potential (str or amuse.potential class): The potential to use. 
                             Defaults to "MWPotential2014".
                             Can be a custom potential object.

    Returns:
        tuple: (final_cluster_particles, simulation_history)
               - final_cluster_particles: The AMUSE Particles set at the end of the simulation.
               - simulation_history: A numpy structured array with time, CoM position, etc.
    """

    # 1. Set up the N-body converter for the cluster's internal dynamics
    # This is crucial for performance and accuracy of the direct N-body code.
    converter = nbody_system.nbody_to_si(cluster_mass, virial_radius)

    # 2. Create the star cluster using a Plummer model
    # The stars are initially centered at (0,0,0) with CoM velocity (0,0,0).
    cluster_particles = new_plummer_model(number_of_stars, convert_nbody=converter)
    cluster_particles.id = np.arange(number_of_stars) + 1 # Assign unique IDs

    # Shift the cluster to its starting position and give it its orbital velocity
    cluster_particles.position += initial_position
    cluster_particles.velocity += initial_velocity
    
    # 3. Initialize the direct N-body gravity code for internal dynamics
    # Hermite is a good general-purpose choice.
    internal_gravity = Hermite(converter)
    internal_gravity.particles.add_particles(cluster_particles)
    
    # We need a channel to copy data back from the code to our script
    channel_from_gravity = internal_gravity.particles.new_channel_to(cluster_particles)

    # 4. Define the external galactic potential
    # MWPotential2014 is a realistic, built-in Milky Way potential.
    # You could also build your own, e.g., MiyamotoNagaiPotential + NFWPotential
    if galaxy_potential == "MWPotential":
        external_potential = MilkyWay_galaxy()
    else:
        # Allows passing a custom potential object
        external_potential = galaxy_potential

    # 5. Set up the Bridge
    # This couples the internal gravity solver with the external potential field.
    # The external_potential will apply a force to every particle in internal_gravity.
    # use_threading=False is often more stable and easier to debug.
    coupled_system = bridge(verbose=False, use_threading=False)
    coupled_system.add_system(internal_gravity, (external_potential,))
    
    # Set the integration timestep for the Bridge. This should be small enough
    # to resolve the cluster's orbit. The internal Hermite code will use its
    # own, smaller, adaptive timesteps.
    coupled_system.timestep = time_step / 4. # Use a smaller bridge step for accuracy

    # 6. Prepare for the simulation loop
    times = np.arange(0, end_time.value_in(units.Myr), time_step.value_in(units.Myr)) | units.Myr
    
    # Create a structured array to store results
    history = np.zeros(len(times), dtype={'names':['time', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'core_radius'], 
                                          'formats':['f8']*8})
    
    print("Starting simulation...")
    print(f"  Evolving cluster of {number_of_stars} stars for {end_time}")

    # 7. The main simulation loop
    for i, t in enumerate(times):
        # Evolve the bridged system
        coupled_system.evolve_model(t)

        # Copy the particle data back from the gravity code to the script's memory
        channel_from_gravity.copy()
        
        # Calculate diagnostics
        com = cluster_particles.center_of_mass()
        com_vel = cluster_particles.center_of_mass_velocity()
        core_radius = cluster_particles.core_radius(convert_nbody=converter)

        # Store data
        history[i]['time'] = coupled_system.model_time.value_in(units.Myr)
        history[i]['x'] = com.x.value_in(units.kpc)
        history[i]['y'] = com.y.value_in(units.kpc)
        history[i]['z'] = com.z.value_in(units.kpc)
        history[i]['vx'] = com_vel.x.value_in(units.kms)
        history[i]['vy'] = com_vel.y.value_in(units.kms)
        history[i]['vz'] = com_vel.z.value_in(units.kms)
        history[i]['core_radius'] = core_radius.value_in(units.parsec)

        print(f"  Time: {t.in_(units.Myr):.2f}, CoM: ({com.x.in_(units.kpc):.2f}, {com.y.in_(units.kpc):.2f}) kpc")

    # 8. Clean up and stop the codes
    print("Simulation finished.")
    internal_gravity.stop()
    
    return cluster_particles, history


# =============================================================================
# Example Usage
# =============================================================================
if __name__ == '__main__':
    
    # --- Simulation Parameters ---
    N_STARS = 500
    CLUSTER_MASS = 1000.0 | units.MSun
    VIRIAL_RADIUS = 1.0 | units.parsec
    
    # Start the cluster at 8.5 kpc on the x-axis (like the Sun)
    # with a circular velocity of 220 km/s in the y-direction.
    INITIAL_POS = [8.5, 0.0, 0.0] | units.kpc
    INITIAL_VEL = [0.0, 220.0, 10.0] | units.kms # Small vz to make orbit 3D
    
    SIM_END_TIME = 250.0 | units.Myr # Evolve for ~1 full orbit
    SIM_TIME_STEP = 1.0 | units.Myr   # Save data every 1 Myr
    
    # --- Run the simulation ---
    final_particles, sim_history = evolve_cluster_in_galaxy(
        number_of_stars=N_STARS,
        cluster_mass=CLUSTER_MASS,
        virial_radius=VIRIAL_RADIUS,
        end_time=SIM_END_TIME,
        time_step=SIM_TIME_STEP,
        initial_position=INITIAL_POS,
        initial_velocity=INITIAL_VEL
    )

    # --- Analysis and Plotting ---
    print("\n--- Final State ---")
    final_com = final_particles.center_of_mass()
    print(f"Final CoM Position: {final_com.in_(units.kpc)}")
    
    # Plot the orbit of the cluster's center of mass
    plt.figure(figsize=(8, 8))
    plt.plot(sim_history['x'], sim_history['y'], label='Cluster Orbit')
    plt.scatter([0], [0], color='yellow', s=100, marker='*', label='Galactic Center')
    plt.scatter([sim_history['x'][0]], [sim_history['y'][0]], color='green', s=50, label='Start')
    plt.scatter([sim_history['x'][-1]], [sim_history['y'][-1]], color='red', s=50, label='End')
    
    plt.xlabel("X [kpc]")
    plt.ylabel("Y [kpc]")
    plt.title("Orbit of Star Cluster in Galactic Potential")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.axis('equal')
    plt.show()

    # Plot the evolution of the core radius
    plt.figure(figsize=(10, 5))
    plt.plot(sim_history['time'], sim_history['core_radius'])
    plt.xlabel("Time [Myr]")
    plt.ylabel("Core Radius [pc]")
    plt.title("Evolution of the Cluster's Core Radius")
    plt.grid(True)
    plt.show()