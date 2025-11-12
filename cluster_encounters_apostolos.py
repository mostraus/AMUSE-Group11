from amuse.lab import *
from amuse.couple import bridge
from amuse.ext.orbital_elements import orbital_elements
from tqdm import tqdm
import numpy as np
import csv
from matplotlib import pyplot as plt

# ========================================
#  STEP 1–2: MILKY WAY + STAR CLUSTER SETUP
# ========================================

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

    def get_potential_at_point(self, eps, x, y , z):
        r = (x**2+y**2+z**2)**0.5
        R = (x**2+y**2)**0.5
        # bulge
        b1 = 0.3873 | units.kpc
        pot_bulge = -constants.G*self.Mb/(r**2+b1**2)**0.5 
        # disk
        a2 = 5.31 | units.kpc
        b2 = 0.25 | units.kpc
        pot_disk = -constants.G*self.Md/(R**2 + (a2+(z**2+b2**2)**0.5)**2)**0.5
        # halo
        a3 = 12.0 | units.kpc
        cut_off = 100 | units.kpc
        d1 = r/a3
        c = 1 + (cut_off/a3)**1.02
        pot_halo = (
            -constants.G*(self.Mh/a3)*d1**1.02/(1+d1**1.02)
            -(constants.G*self.Mh/(1.02*a3))
            *(-1.02/c + np.log(c) + 1.02/(1+d1**1.02) - np.log(1.0 + d1**1.02))
        )
        return 2*(pot_bulge + pot_disk + pot_halo)

    def get_gravity_at_point(self, eps, x, y, z): 
        r = (x**2+y**2+z**2)**0.5
        R = (x**2+y**2)**0.5
        b1 = 0.3873 | units.kpc
        a2 = 5.31 | units.kpc
        b2 = 0.25 | units.kpc
        a3 = 12.0 | units.kpc

        # Forces
        force_bulge = -constants.G*self.Mb/(r**2+b1**2)**1.5
        d = a2 + (z**2+b2**2)**0.5
        force_disk = -constants.G*self.Md/(R**2+d**2)**1.5
        d1 = r/a3
        force_halo = -constants.G*self.Mh*d1**0.02/(a3**2*(1+d1**1.02))

        ax = force_bulge*x + force_disk*x + force_halo*x/r
        ay = force_bulge*y + force_disk*y + force_halo*y/r
        az = force_bulge*z + force_disk*d*z/(z**2+b2**2)**0.5 + force_halo*z/r

        return ax, ay, az


# ---------- Create star cluster ----------
N = 100
masses = new_salpeter_mass_distribution(N, 1 | units.MSun, 100 | units.MSun)
R_plummer = 10 | units.parsec
M_total = masses.sum()

converter = nbody_system.nbody_to_si(M_total, R_plummer)
cluster = new_plummer_model(N, convert_nbody=converter)
cluster.mass = masses

# Place cluster at 8.5 kpc and give orbital velocity
cluster.position += (8.5, 0, 0) | units.kpc
cluster.velocity += (0, 22, 0) | units.kms

# ---------- Set up dynamics ----------
gravity_code = Huayno(converter)
gravity_code.particles.add_particles(cluster)
ch_g2l = gravity_code.particles.new_channel_to(cluster)

MWG = MilkyWay_galaxy()

gravity = bridge.Bridge(use_threading=False)
gravity.add_system(gravity_code, (MWG,))
gravity.timestep = 1 | units.Myr

# ========================================
#  STEP 3–4: ENCOUNTER DETECTION & DATABASE
# ========================================

def enclosed_mass_MW(R, MWG):
    """
    Approximate enclosed mass of the Milky Way within radius R.
    Returns mass with AMUSE units.
    """
    R = R.as_quantity_in(units.kpc)  # ensure AMUSE length quantity
    Mb, Md, Mh = MWG.Mb, MWG.Md, MWG.Mh
    a3 = 12.0 | units.kpc
    R_d = 5.0 | units.kpc

    # Compute smooth, dimensionless ratios
    f_disk = (R / (R + R_d))
    f_halo = (R / (R + a3))

    M_enclosed = Mb + Md * f_disk + Mh * f_halo
    return M_enclosed


# Storage for results
times = np.arange(0., 1000, 1.) | units.Myr
x_cm = [] | units.kpc
y_cm = [] | units.kpc
encounters = []

for time in tqdm(times):
    gravity.evolve_model(time)
    ch_g2l.copy()

    # Record cluster centre of mass
    cm = cluster.center_of_mass()
    x_cm.append(cm.x)
    y_cm.append(cm.y)

    # Galactic distance and enclosed mass
    R_gal = (cm.x**2 + cm.y**2 + cm.z**2)**0.5
    M_enc = enclosed_mass_MW(R_gal, MWG)

    # Compute Hill radii
    hill_radii = R_gal * (cluster.mass / (3 * M_enc))**(1/3)

    # Check for close encounters
    pos = cluster.position
    vel = cluster.velocity

    for i in range(N):
        for j in range(i + 1, N):
            rij = (pos[i] - pos[j]).length()
            # limit = 2 * max(hill_radii[i], hill_radii[j])
            limit = 1 | units.pc
            if rij < limit:
                v_rel = (vel[i] - vel[j]).length()
                encounters.append([
                    time.value_in(units.Myr),
                    i,
                    j,
                    rij.value_in(units.AU),
                    hill_radii[i].value_in(units.AU),
                    hill_radii[j].value_in(units.AU),
                    v_rel.value_in(units.kms)
                ])

    # print(f'Encounters detected at {time.in_(units.Myr)}: {len(encounters)}')

gravity.stop()

# ---------- Save encounters to CSV ----------
with open("encounter_database.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "time_Myr", "star_i", "star_j",
        "distance_AU", "Hill_r_i_AU", "Hill_r_j_AU", "v_rel_kms"
    ])
    writer.writerows(encounters)

print(f"\nTotal encounters recorded: {len(encounters)}")
if encounters:
    print("First few encounters:")
    for row in encounters[:5]:
        print(row)

# ========================================
#  PLOT CLUSTER ORBIT (OPTIONAL)
# ========================================

plt.figure(figsize=(6,6))
plt.plot(x_cm.value_in(units.kpc), y_cm.value_in(units.kpc), lw=1, label='Cluster CoM')
plt.scatter(x_cm[0].value_in(units.kpc), y_cm[0].value_in(units.kpc), s=10)
plt.xlabel("x [kpc]")
plt.ylabel("y [kpc]")
plt.gca().set_aspect("equal", adjustable="box")
plt.legend()
plt.title(f"Centre-of-Mass Orbit of {N}-Star Cluster in Milky Way Potential")
plt.show()
