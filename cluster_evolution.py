import numpy as np
from amuse.units import units, constants
from amuse.lab import nbody_system
from amuse.couple import bridge
from amuse.community.huayno import Huayno
from amuse.ic.salpeter import new_salpeter_mass_distribution
from amuse.ic.fractalcluster import new_fractal_cluster_model
from amuse.lab import Particle, Particles, units, nbody_system
from scipy import cluster
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import math, numpy
from amuse.lab import *
from amuse.community.ph4.interface import ph4
import csv
import matplotlib.animation as animation
from IPython.display import HTML

# Define the Milky Way potential with a bulge, disk, and halo - borrowed from Bovy 2015

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
    

def starting_conditions_cluster(number_of_stars,velocity_x_direction, velocity_y_direction,velocity_z_direction ,position_x,position_y, position_z, encounter_radious, seed=None):
    """
    Generate a fractal star cluster with a Salpeter mass distribution. With specified initial position and velocity.
    Setting the star radious as encounter_radious for detection purposes.
    Returns
    -------
    cluster : ParticleSet
        The generated star cluster.
    converter : nbody_system.nbody_to_si
        The converter used for unit conversions.
    """
    if seed is not None:
        numpy.random.seed(seed)
    masses = new_salpeter_mass_distribution(number_of_stars, 1|units.MSun, 30|units.MSun)
    converter = nbody_system.nbody_to_si(masses.sum(), 1|units.parsec)
    cluster = new_fractal_cluster_model(number_of_stars, converter)
    cluster.mass = masses
    cluster.move_to_center()
    cluster.position += [position_x, position_y, position_z] | units.kpc
    cluster.velocity += [velocity_x_direction, velocity_y_direction, velocity_z_direction] | units.kms
    cluster.id = range(len(cluster))

    #Remember this is not meaning the physical radious of the stars, but for encounter detection purposes
    cluster.radius = encounter_radious

    cluster_star_mass = []
    for i in range(len(cluster)):
        cluster_star_mass.append(cluster[i].mass.value_in(units.MSun))

    return cluster, converter


def plot_the_starting_cluster_position_xy(cluster):
    """
    Plot the initial positions of the star cluster in the XY plane.
    Parameters
    ----------
    cluster : ParticleSet
        The star cluster to be plotted.
    """
    x_start = cluster.x.value_in(units.parsec)
    y_start = cluster.y.value_in(units.parsec)
    z_start = cluster.z.value_in(units.parsec)

    plt.figure(figsize=(7, 7))
    plt.scatter(cluster.x.value_in(units.kpc), cluster.y.value_in(units.kpc), c='blue', s=10)
    plt.xlabel("X [kpc]")
    plt.ylabel("Y [kpc]")
    plt.show()


def save_initial_state_of_cluster(cluster):
    """
    Save the initial state of the star cluster, including positions, velocities, masses, and minimal distances to other stars.
    Parameters
    ----------
    cluster : ParticleSet
        The star cluster whose initial state is to be saved.
    Returns
    -------
    df_initial : pandas.DataFrame
        DataFrame containing the initial state information of the cluster.
    """

    initial_state_list = []

    for particle in cluster:
        star_min = []
        for particle2 in cluster:
            dis = ((particle.x.value_in(units.parsec) - particle2.x.value_in(units.parsec))**2 + (particle.y.value_in(units.parsec) - particle2.y.value_in(units.parsec))**2 + (particle.z.value_in(units.parsec) - particle2.z.value_in(units.parsec))**2)**0.5
            if dis > 0:
                star_min.append(dis)
        min(star_min) | units.parsec
        initial_state_list.append({
            "key": particle.key,  # The persistent ID
            "x0": particle.x.value_in(units.parsec),
            "y0": particle.y.value_in(units.parsec),
            "z0": particle.z.value_in(units.parsec),
            "vx0": particle.vx.value_in(units.kms),
            "vy0": particle.vy.value_in(units.kms),
            "vz0": particle.vz.value_in(units.kms),
            "mass": particle.mass.value_in(units.MSun),
            "minimal_distance_to_other_star": min(star_min)
        })

    return initial_state_list
    

def get_internal_energy(cluster):
    """
    Calculate and return the energy of a cluster as all. 
    Parameters
    ----------
    cluster : ParticleSet
        The star cluster whose energy components are to be calculated.
    Returns
    -------
    K_internal : quantity
        Internal kinetic energy of the cluster.
    U_internal : quantity
        Internal potential energy of the cluster.
    Q_internal : float
        Virial ratio of the cluster (K/|U|).
    E_internal : quantity
        Total internal energy of the cluster (K + U).
    """

    # Save original velocities so we don't break the simulation
    original_velocities = cluster.velocity.copy()

    # 1. Calculate Center of Mass Velocity
    v_cm = cluster.center_of_mass_velocity()

    # 2. Shift to Center of Mass Frame (Remove bulk motion)
    cluster.velocity -= v_cm

    # 3. Calculate Energy of the cluster internals
    K_internal = cluster.kinetic_energy()
    U_internal = cluster.potential_energy()
    Q_internal = K_internal / abs(U_internal)
    E_internal = K_internal + U_internal

    # 4. RESTORE velocities (Critical! Otherwise your simulation stops moving!)
    cluster.velocity = original_velocities
    return K_internal, U_internal, Q_internal, E_internal


def run_evolution(t_end, dt, t0, cluster, galaxy, limit):
    """
    Run the evolution of the star cluster in the Milky Way potential, tracking various properties over time.
    Parameters
    ----------
    t_end : quantity
        The end time for the simulation. Remember this is a time of gravity code, so it can be a bit different from the bridge time.
    dt : quantity
        The time step for the simulation. It is maximum time step if the code is not interrupted by collisions.
    t0 : quantity
        The starting time for the simulation.
    cluster : ParticleSet
        The star cluster to be evolved.
    galaxy : MilkyWay_galaxy
        The Milky Way potential in which the cluster evolves.
    limit : quantity
        The minimum radius limit for stars after collision. Belove thus limit we assume they merge and set radius to 0.
    Returns
    -------
    """

    t_history = []
    x_history = []
    y_history = []
    z_history = []
    center_of_mass_history = []
    bound_history = []
    bound_mass_history = []


    energy_overtime = []
    Q_overtime = []
    t_energy = []

    #initialize the n-body code
    t = t0
    gravity_code = ph4(converter, number_of_workers=1)
    gravity_code.particles.add_particles(cluster)

    channel_from_gd_to_framework = gravity_code.particles.new_channel_to(cluster)

    #setting the stopping condition for collisions
    stopping_condition = gravity_code.stopping_conditions.collision_detection
    stopping_condition.enable()

    #initialize the bridge to conect the galaxy potential with the cluster
    gravity = bridge.Bridge(use_threading=False)
    gravity.add_system(gravity_code, (galaxy,) )
    gravity.evolve_model(t+dt)

    
    interactions_list =[]
    i = 0                     

    while t < t_end:
        if stopping_condition.is_set():
            # print(gravity.model_time.in_(units.yr))
            t = gravity_code.model_time.in_(units.yr)
            print(gravity_code.model_time.in_(units.yr))

            #The real time always gravity_code.model_time
            t_history.append(gravity_code.model_time.in_(units.yr))
            x_history.append(cluster.x.value_in(units.parsec))
            y_history.append(cluster.y.value_in(units.parsec))
            z_history.append(cluster.z.value_in(units.parsec))

            center_of_mass = cluster.center_of_mass()
            center_of_mass_history.append(center_of_mass)
            
        
            # Calculate Energy correctly (subtracting bulk motion just for math)
            K_internal, U_internal, Q_internal, E_internal= get_internal_energy(cluster)
            
            t_energy.append(gravity_code.model_time.in_(units.yr))
            energy_overtime.append(E_internal.value_in(units.erg))
            Q_overtime.append(Q_internal)

            # Handle the collision and saving the data about it
            part_set_1 = stopping_condition.particles(0)
                
            part_set_2 = stopping_condition.particles(1)
            
            for p1, p2 in zip(part_set_1, part_set_2):
                #to solve that they will be stopping instantly
                p1.radius = p1.radius/2
                p2.radius = p2.radius/2

                # If they are to close we assume they merge and set their radious to 0
                if p1.radius < limit:
                    p1.radius = 0 | units.AU
                if p2.radius < limit:
                    p2.radius = 0 | units.AU

                # Calculate mass ratio
                if p1.mass < p2.mass:
                    mass_ratio = p2.mass / p1.mass
                else:
                    mass_ratio = p1.mass / p2.mass


            
            interactions_list.append((gravity_code.model_time.in_(units.yr), stopping_condition.particles(0).key, stopping_condition.particles(1).key, p1.radius.in_(units.AU), p2.radius.in_(units.AU),p1.x.in_(units.kpc),p1.y.in_(units.kpc),p1.z.in_(units.kpc),p2.x.in_(units.kpc),p2.y.in_(units.kpc),p2.z.in_(units.kpc),p1.vx.in_(units.kms),p1.vy.in_(units.kms),p1.vz.in_(units.kms),p2.vx.in_(units.kms),p2.vy.in_(units.kms),p2.vz.in_(units.kms),p1.mass.in_(units.MSun),p2.mass.in_(units.MSun),mass_ratio))
            #need to initialize again after the stopping condition
            gravity.evolve_model(t+dt)
            i += 1
        else:
            #If no collision just evolve normally
            t += dt
            t_history.append(gravity_code.model_time.in_(units.yr))
            x_history.append(cluster.x.value_in(units.parsec))
            y_history.append(cluster.y.value_in(units.parsec))
            z_history.append(cluster.z.value_in(units.parsec))
            
            #Here we also need to evolve the gravity
            gravity.evolve_model(t)
            channel_from_gd_to_framework.copy()
            
            # Calculate Energy correctly (subtracting bulk motion just for math)
            K_internal, U_internal, Q_internal, E_internal = get_internal_energy(cluster)

            t_energy.append(gravity_code.model_time.in_(units.yr))
            energy_overtime.append(E_internal.value_in(units.erg))
            Q_overtime.append(Q_internal)
            #If a whole time step with no collisions has passed - all stars come back to the normal radious
            cluster.radius = 500 | units.AU

    print("end",gravity_code.model_time.in_(units.Myr))
    gravity.stop()
    #print(interactions_list)
    return (interactions_list, t_history, x_history, y_history, z_history, center_of_mass_history, bound_history, bound_mass_history, t_energy, energy_overtime, Q_overtime, cluster)


def save_file_with_interactions(interactions_list, filename):
    """
    Save the interactions list to a CSV file.
    Parameters
    ----------
    interactions_list : list
        List of interactions to be saved.
    filename : str
        Name of the output CSV file.
    """
    
    clean_rows = []
    for row_tuple in interactions_list:
        cleaned_row = []
        for item in row_tuple:
            # Check if the item is an AMUSE Quantity (has a unit attached)
            if hasattr(item, "number"):
                # .number strips the unit (since you already did .in_() in the main loop)
                cleaned_row.append(item.number)
            
            # Check if the item is a numpy array (like the keys/IDs usually are)
            elif hasattr(item, "__len__") and not isinstance(item, str):
                # Take the first item if it's a list/array (e.g. key [102] -> 102)
                cleaned_row.append(item[0])
                
            # Fallback for standard python types
            else:
                cleaned_row.append(item)
                
        clean_rows.append(cleaned_row)

    # 2. Write to CSV
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["time (yr)", "particle1_id", "particle2_id", "particle1_radius (AU)", "particle2_radius (AU)", "particle1_x (kpc)","particle1_y (kpc)","particle1_z (kpc)","particle2_x (kpc)","particle2_y (kpc)","particle2_z (kpc)","particle1_vx (km/s)","particle1_vy (km/s)","particle1_vz (km/s)","particle2_vx (km/s)","particle2_vy (km/s)","particle2_vz (km/s)","particle1_mass (MSun)","particle2_mass (MSun)","mass_ratio", "total_energy (erg)", "virial_ratio (Q)"])
        # Write data
        writer.writerows(clean_rows)
    return clean_rows


def plot_trajectory_xy(x_history, y_history):
    """
    Plot the trajectories of all stars in the XY plane.
    Parameters
    ----------
    x_history : list
        List of x positions over time for all stars.

    y_history : list
        List of y positions over time for all stars.
    """
    x_all = numpy.array(x_history).T        # Shape: (stars, steps)
    y_all = numpy.array(y_history).T        # Shape: (stars, steps)

    plt.figure(figsize=(8, 8))

    for i in range(x_all.shape[0]):
        plt.plot(x_all[i], y_all[i], linewidth=0.5, alpha=0.6)


    plt.xlabel('X Position [parsec]')
    plt.ylabel('Y Position [parsec]')
    plt.title(f'Trajectories of All {x_all.shape[0]} Stars')

    plt.axis('equal')

    plt.show()
    

def animate_trajectory_xy(x_history, y_history, filename):
    """
    Create an animation of the trajectories of all stars in the XY plane.
    Parameters
    ----------
    x_history : list
        List of x positions over time for all stars.
    y_history : list
        List of y positions over time for all stars.
    """
        # --- SETTINGS ---
    # Skip frames to make generation faster (e.g., 5 means plot every 5th time step)
    #skip_step = 5 
    # Total available steps in your data
    x_all = numpy.array(x_history).T        # Shape: (stars, steps)
    y_all = numpy.array(y_history).T        # Shape: (stars, steps)

    total_data_steps = x_all.shape[1] 

    fig, ax = plt.subplots(figsize=(7, 7))

    lines = []

    # 1. Initialize the lines
    for i in range(number_of_stars):
        # Initialize with empty data
        # Note: ax.plot returns a list, so we unwrap it with comma: line, = ...
        line, = ax.plot([], [], lw=1, alpha=0.7, label=f"Star {i}")
        lines.append(line)

    # 2. Set Axis Limits Automatically based on the data range
    # Adding a 10% margin so stars don't hit the edge
    x_min, x_max = x_all.min(), x_all.max()
    y_min, y_max = y_all.min(), y_all.max()
    margin_x = (x_max - x_min) * 0.1
    margin_y = (y_max - y_min) * 0.1

    ax.set_xlim(x_min - margin_x, x_max + margin_x)
    ax.set_ylim(y_min - margin_y, y_max + margin_y)
    # ax.set_xlim(8450,8550)
    # ax.set_ylim(-20,20)

    ax.set_xlabel("x [pc]")
    ax.set_ylabel("y [pc]")
    ax.set_title(f"Cluster Evolution ({t_end})")

    # 3. Update Function
    def update(frame):
        # frame is the index of the time step
        for i in range(number_of_stars):
            # Slice from start (0) to current frame to show the full trail
            # x_all[star_index, 0:current_time_index]
            x = x_all[i, :frame]
            y = y_all[i, :frame]
            
            lines[i].set_data(x, y)
        return lines 

    # 4. Generate Animation
    frames=range(0, total_data_steps) #makes it lighter/faster to generate
    ani = animation.FuncAnimation(
        fig=fig, 
        func=update, 
        frames=frames, 
        interval=30, 
        blit=True
    )

    plt.close() # Prevents showing the static plot underneath
    HTML(ani.to_jshtml())
    ani.save(filename, writer="ffmpeg", fps=30)
    
def start_analyzis_get_pandas_dataframe():
    data = pd.DataFrame(clean_rows, columns=["time (yr)", "particle1_id", "particle2_id", "particle1_radius (AU)", "particle2_radius (AU)", "particle1_x (kpc)","particle1_y (kpc)","particle1_z (kpc)","particle2_x (kpc)","particle2_y (kpc)","particle2_z (kpc)","particle1_vx (km/s)","particle1_vy (km/s)","particle1_vz (km/s)","particle2_vx (km/s)","particle2_vy (km/s)","particle2_vz (km/s)","particle1_mass (MSun)","particle2_mass (MSun)","mass_ratio"])
    return data


def how_many_stars_have_interaction(data, thresholds, number_of_stars):
    # 1. Define thresholds
    thresholds = [1000, 500, 250, 125, 75, 38, ]
    results = {}

    for limit in thresholds:

        
        mask = (data["particle1_radius (AU)"] + data["particle2_radius (AU)"]) <= (limit + 1e-3) # slight float buffer
        subset = data[mask]
        
        # B. Get unique stars involved in these specific interactions
        involved_stars = set(pd.concat([subset["particle1_id"], subset["particle2_id"]]).unique())
        
        # C. Calculate Percentage
        count = len(involved_stars)
        pct = (count / number_of_stars) * 100
        results[limit] = pct

    # 3. Print the "Funnel" Report
    print("-" * 40)
    print(f"Total Stars in Cluster: {number_of_stars}")

    print("-" * 40)
    for limit in thresholds:
        print(f"Stars with interaction <= {limit} AU: {results[limit]:.2f}%")
    print("-" * 40)
    return results


def print_interesting_interactions(data):
    df_pairs = pd.concat([
        data[["particle1_id", "particle2_id"]].rename(columns={"particle1_id": "Star", "particle2_id": "Partner"}),
        data[["particle2_id", "particle1_id"]].rename(columns={"particle2_id": "Star", "particle1_id": "Partner"})
    ])

    interaction_counts = df_pairs.groupby("Star")["Partner"].nunique()

    # Find the star with the maximum count
    most_active_star_id = interaction_counts.idxmax()
    max_interactions = interaction_counts.max()

    print(f"Star with most unique interactions: ID {int(most_active_star_id)}")
    print(f"Number of unique partners: {max_interactions}")

    # # Optional: Print the list of partners for that star
    # partners = df_pairs[df_pairs["Star"] == most_active_star_id]["Partner"].unique()
    # print(f"Partners: {sorted(partners)}")

    max_ratio_index = data["mass_ratio"].idxmax()
    max_ratio_row = data.loc[max_ratio_index]

    star1 = int(max_ratio_row["particle1_id"])
    star2 = int(max_ratio_row["particle2_id"])
    max_ratio = max_ratio_row["mass_ratio"]

    print("-" * 30)
    print(f"Biggest Mass Ratio: {max_ratio:.4f}")
    print(f"Stars involved: ID {star1} and ID {star2}")


            
def plot_initial_and_ending_position_color_by_interactions(data, initial_state_list):
    '''
    Create the plots showing stariting and ending position of a stars in a cluster. 
    ----------
    data : dataframe
        Data fram with a stars which had an interactions
    initial_state_list : list
        Starting positions, keys and velocities of a stars in a cluster
    '''


    union_particles = set(data['particle1_id']).union(set(data['particle2_id']))
    #print(len(union_particles))
    v_cm = cluster.center_of_mass_velocity()
    unaffected_x, unaffected_y = [], []
    affected_x, affected_y = [], []
    unaffected_mass, affected_mass = [], []
    unaffected_velocity, affected_velocity = [], []
    unaffected_vx, affected_vx = [], []
    unaffected_vy, affected_vy = [], []
    unaffected_vz, affected_vz = [], []
    unaffected_x_end, unaffected_y_end = [], []
    affected_x_end, affected_y_end = [], []
    unaffected_mass_end, affected_mass_end = [], []
    affected_velocity_end, unaffected_velocity_end = [], []
    affected_vx_end, unaffected_vx_end = [], []
    affected_vy_end, unaffected_vy_end = [], []
    affected_vz_end, unaffected_vz_end = [], []


    for star in initial_state_list:
        if star['key'] in union_particles:
            affected_x.append(star['x0'])
            affected_y.append(star['y0'])
            affected_mass.append(star['mass'] * 2)
            
            affected_vx.append(star['vx0'])
            affected_vy.append(star['vy0'])
            affected_vz.append(star['vz0'])

            velocity = (star['vx0']**2 + star['vy0']**2 + star['vz0']**2)**0.5
            affected_velocity.append(velocity)


        else:
            unaffected_x.append(star['x0'])
            unaffected_y.append(star['y0'])
            unaffected_mass.append(star['mass'] * 2)
            
            unaffected_vx.append(star['vx0'])
            unaffected_vy.append(star['vy0'])
            unaffected_vz.append(star['vz0'])
            velocity = (star['vx0']**2 + star['vy0']**2 + star['vz0']**2)**0.5
            unaffected_velocity.append(velocity)


    plt.figure(figsize=(8, 8))
    plt.scatter(unaffected_x, unaffected_y, c='blue', s=unaffected_mass, label=f'Unaffected {len(unaffected_x)}')
    plt.scatter(affected_x, affected_y, c='red', s=affected_mass, alpha=0.7, label=f'Affected {len(affected_x)}')
    plt.xlabel("Initial X Position [pc]")
    plt.ylabel("Initial Y Position [pc]")
    plt.title("Initial Cluster Positions by Interaction Status - size of point depend on mass")
    plt.legend()
    plt.show()

    print(np.median(unaffected_mass), np.median(affected_mass))
    print(np.mean(unaffected_mass), np.mean(affected_mass))

        # 2. Iterate
    for i in range(len(cluster)):
        
        # Calculate relative velocity components (Star Velocity - Cluster Bulk Velocity)
        dvx = cluster.vx[i] - v_cm[0]
        dvy = cluster.vy[i] - v_cm[1]
        dvz = cluster.vz[i] - v_cm[2]
        
        # Calculate magnitude of this relative velocity vector
        velocity_mag = (dvx**2 + dvy**2 + dvz**2)**0.5
        
        # Visualization scaling: Multiply by 10 or 20 to make the dots visible
        # otherwise a star with 0.5 km/s is invisible (0.5 pixel size)
        velocity_mag = velocity_mag.value_in(units.kms) * 10
        

        if cluster.key[i] in union_particles:
            affected_x_end.append(cluster.x[i].value_in(units.pc))
            affected_y_end.append(cluster.y[i].value_in(units.pc))
            affected_mass_end.append(cluster.mass[i].value_in(units.MSun))
            affected_velocity_end.append(velocity_mag)
            affected_vx_end.append(cluster.vx[i].value_in(units.kms))
            affected_vy_end.append(cluster.vy[i].value_in(units.kms))
            affected_vz_end.append(cluster.vz[i].value_in(units.kms))
        else:
            unaffected_x_end.append(cluster.x[i].value_in(units.pc))
            unaffected_y_end.append(cluster.y[i].value_in(units.pc))
            unaffected_mass_end.append(cluster.mass[i].value_in(units.MSun))
            unaffected_velocity_end.append(velocity_mag)
            unaffected_vx_end.append(cluster.vx[i].value_in(units.kms))
            unaffected_vy_end.append(cluster.vy[i].value_in(units.kms))
            unaffected_vz_end.append(cluster.vz[i].value_in(units.kms))

    plt.figure(figsize=(8, 8))

    # Added alpha to blue dots so we can see overlaps better
    plt.scatter(unaffected_x_end, unaffected_y_end, c='blue', s=unaffected_mass_end, label=f'Unaffected {len(unaffected_x_end)}')
    plt.scatter(affected_x_end, affected_y_end, c='red', s=affected_mass_end, label=f'Affected {len(affected_x_end)}')

    plt.xlabel("Ending X Position [pc]")
    plt.ylabel("Ending Y Position [pc]")
    plt.title("Cluster Positions in the end by Interaction Status -size of dots depend on mass")
    plt.legend()
    plt.axis('equal') # Important for spatial plots
    plt.show()


def statiscit_what_initial_condition_set_to_have_interaction(data, initial_state_list):
    """
    Analyze which initial stellar conditions are associated with having at least
    one interaction during the simulation.

    This function separates stars into two groups:
    - *Affected*: stars that appear in at least one interaction (present in
      `particle1_id` or `particle2_id` in `data`)
    - *Unaffected*: stars that never appear in any interaction

    For both groups, it computes and compares summary statistics (median, mean,
    standard deviation) of several initial-condition properties, including:
    - Velocity magnitude (with bulk motion subtracted from vy)
    - Distance to the system center
    - Stellar mass
    - Minimal initial distance to another star

    The results are printed to stdout and are intended for exploratory,
    diagnostic analysis rather than as return values.

    Parameters
    ----------
    data : pandas.DataFrame
        Interaction table containing at least the columns:
        - 'particle1_id'
        - 'particle2_id'
        Each row represents an interaction between two stars.

    initial_state_list : list of dict
        List of dictionaries describing the initial state of each star.
        Each dictionary must contain the keys:
        - 'key' : unique star identifier (must match particle IDs in `data`)
        - 'x0', 'y0', 'z0' : initial positions
        - 'vx0', 'vy0', 'vz0' : initial velocities
        - 'mass' : stellar mass
        - 'minimal_distance_to_other_star' : closest initial separation to
          any other star

    Returns
    -------
    None
        All results are printed to standard output.
    """


    union_particles = set(data['particle1_id']).union(set(data['particle2_id']))
    print(len(union_particles))

    unaffected_x, unaffected_y = [], []
    affected_x, affected_y = [], []
    unaffected_mass, affected_mass = [], []
    unaffected_velocity, affected_velocity = [], []
    unaffected_vx, affected_vx = [], []
    unaffected_vy, affected_vy = [], []
    unaffected_vz, affected_vz = [], []
    unaffected_distance_to_center, affected_distance_to_center = [], []
    unaffected_distance_to_other_star, affected_distance_to_other_star = [], []

    for star in initial_state_list:
        if star['key'] in union_particles:
            affected_x.append(star['x0'])
            affected_y.append(star['y0'])
            affected_mass.append(star['mass'] * 2)
            
            vy = star['vy0'] - 40  # Subtracting bulk motion for analysis
            affected_vx.append(star['vx0'])
            affected_vy.append(vy)
            affected_vz.append(star['vz0'])

            velocity = (star['vx0']**2 + vy**2 + star['vz0']**2)**0.5
            affected_velocity.append(velocity)

            dis_center = ((star['x0']- 8500)**2 + star['y0']**2 + star['z0']**2)**0.5
            affected_distance_to_center.append(dis_center)

            affected_distance_to_other_star.append(star['minimal_distance_to_other_star'])

        else:
            unaffected_x.append(star['x0'])
            unaffected_y.append(star['y0'])
            unaffected_mass.append(star['mass'] * 2)
            
            vy = star['vy0'] - 40  # Subtracting bulk motion for analysis
            unaffected_vx.append(star['vx0'])
            unaffected_vy.append(vy)
            unaffected_vz.append(star['vz0'])
            velocity = (star['vx0']**2 + vy**2 + star['vz0']**2)**0.5
            unaffected_velocity.append(velocity)
            dis_center = ((star['x0']- 8500)**2 + star['y0']**2 + star['z0']**2)**0.5
            unaffected_distance_to_center.append(dis_center)
            unaffected_distance_to_other_star.append(star['minimal_distance_to_other_star'])


    #What can be a predictior of being affected? Initial position, mass, velocity?
    print("median, mean and standard deviation")

    print("affected_velocity", np.median(affected_velocity), np.mean(affected_velocity), np.std(affected_velocity))
    print("unaffected_velocity", np.median(unaffected_velocity), np.mean(unaffected_velocity), np.std(unaffected_velocity))

    print("affected_distance_to_center", np.median(affected_distance_to_center), np.mean(affected_distance_to_center), np.std(affected_distance_to_center))
    print("unaffected_distance_to_center", np.median(unaffected_distance_to_center), np.mean(unaffected_distance_to_center), np.std(unaffected_distance_to_center))


    print("affected_mass", np.median(affected_mass), np.mean(affected_mass), np.std(affected_mass))
    print("unaffected_mass", np.median(unaffected_mass), np.mean(unaffected_mass), np.std(unaffected_mass))

    print("affected_distance_to_other_star", np.median(affected_distance_to_other_star), np.mean(affected_distance_to_other_star), np.std(affected_distance_to_other_star))
    print("unaffected_distance_to_other_star", np.median(unaffected_distance_to_other_star), np.mean(unaffected_distance_to_other_star), np.std(unaffected_distance_to_other_star))

def how_many_pairs_during_time(data, time1, time2):    
    """
    Analyze how many unique star pairs have interactions over time.
    Parameters  
    ----------
    data : pandas.DataFrame
        Interaction table containing at least the columns:
        - 'time (yr)'
        - 'particle1_id'
        - 'particle2_id'
        Each row represents an interaction between two stars.
    Returns
    -------
    time_series : list of tuples
        Each tuple contains (time, unique_pair_count) representing the number of unique star pairs that have interacted up to that time.
    """
    p1 = data["particle1_id"].values
    p2 = data["particle2_id"].values

    
    interactions = pd.DataFrame({
        'id_min': np.minimum(p1, p2),
        'id_max': np.maximum(p1, p2),
        'time': data['time (yr)']
    })

    first_encounters = interactions.groupby(['id_min', 'id_max'])['time'].min()

    # 3. Obliczenie statystyk
    total_unique_pairs = len(first_encounters)

    if total_unique_pairs > 0:
        # 1 Myr = 1,000,000 lat
        count_1Myr = (first_encounters <= time1).sum()
        
        # 10 Myr = 10,000,000 lat
        count_10Myr = (first_encounters <= time2).sum()
        fraction_1Myr = count_1Myr / total_unique_pairs
        fraction_10Myr = count_10Myr / total_unique_pairs

        print(f"Total unique pairs: {total_unique_pairs}")
        print(f"Interactions within {time1/1e6} Myr: {fraction_1Myr:.2%} ({count_1Myr})")
        print(f"Interactions within {time2/1e6} Myr: {fraction_10Myr:.2%} ({count_10Myr})")
    else:
        print("No interactions in the data.")    



if __name__ == "__main__":
    number_of_stars = 100
    velocity_x = 0 
    velocity_y = 220 
    velocity_z = 0 
    position_x = 8.5 
    position_y = 0 
    position_z = 0
    encounter_radious = 500 | units.AU
    t_end = 100 | units.Myr
    dt = 0.1 | units.Myr
    MWG = MilkyWay_galaxy()
    t = 0 | units.Myr
    limit = 0.001 |units.AU
    name = "interactions.csv"

    seed = 109

    thresholds = [1000, 500, 250, 125, 75, 38]

    time1 = 1e6  # 1 Myr in years
    time2 = 1e7  # 10 Myr in years



    cluster, converter = starting_conditions_cluster(number_of_stars, velocity_x, velocity_y, velocity_z, position_x, position_y, position_z, encounter_radious, seed)
    
    #write_set_to_file(cluster, "cluster.csv", "csv")
    cluster = read_set_from_file("cluster1.csv", "csv", copy_history = False, close_file = False)


    plot_the_starting_cluster_position_xy(cluster)
    df_initial = save_initial_state_of_cluster(cluster)

    K, U, Q, E = get_internal_energy(cluster)


    interactions_list, t_history, x_history, y_history, z_history, center_of_mass_history, bound_history, bound_mass_history, t_energy, energy_overtime, Q_overtime, final_cluster = run_evolution(t_end, dt, t, cluster, MWG, limit)

    clean_rows = save_file_with_interactions(interactions_list, name)

    plot_trajectory_xy(x_history, y_history)

    ## It is taking a few minutes to generate the animation
    #animate_trajectory_xy(x_history, y_history, "cluster_evolution.mp4")

    #to get data you neet to first run the simulation and save the interactions.csv file
    data = start_analyzis_get_pandas_dataframe()

    #Thershols - the list of radious limits to check  - remember the number is a sum of radious of 2 stars
    
    number_of_stars_with_interaction = how_many_stars_have_interaction(data, thresholds, number_of_stars)

    print_interesting_interactions(data)

    plot_initial_and_ending_position_color_by_interactions(data, df_initial)
    statiscit_what_initial_condition_set_to_have_interaction(data, df_initial)

    how_many_pairs_during_time(data, time1, time2)