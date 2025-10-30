from amuse.lab import *


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