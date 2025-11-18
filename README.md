# AMUSE-Group11
Code and Content regarding the AMUSE Project from Group 11

## Topic: The survival of protoplanetary disks in galaxy tidal streams
**Motivation**: Star clusters and galaxies are tidally stripped; some stars with protoplanetary disk may end up in tidal streams.

**Question**: Can protoplanetary disk around stripped stars survive the tidal disruption and ejection into the Galactic halo?

**AMUSE use**: N-body for star cluster and galaxy, hydro for interactions in galaxy potential and encounters.

**Deliverables**: Survival of disks, number of encounters, distribution of disk particles during encounter.

**Why interesting**: Connects disk survival to galactic dynamics.

## Notes
- Run stage 1 of LonelyPlanets 
- Star cluster in potential of galaxy
- Build database of encounters
- Choose interesting star
- Stage 2: protoplanetary disk + 2 stars
- Just hydro code there (it has some gravity already involed)
- How is a protoplanetary disk perturbed by stellar cluster?

**Details**:
- write script for star cluster ($N=1000$?)
- bridge star cluster's evolution with tidal tail of galaxy (gravity + hydro)
- check only if stars come closer than $2~R_\text{Hill}$ (check if it comes within perturbing distance)
- build database of encounters
- calculate disk during encounter, then freeze until next (fast forward)
- write LonelyPlanets ourselves, save close encounters of certain stars (maybe 10?) with stopping condition
- no bridging, just hydro code (can do self-gravity too, just "sloppier")
- during encounter hydro, then check if disk is ok, jump to next encounter
- if there's a problem with star getting bound, just skip it

- **Desired code**:
  - **Stage 1**: run star cluster evolution
  - **Stage 2**: do hydro calculations of encounters
 
- **Desired plot**: distribution of particles of 1 interesting star as function of time during encounter 

## Code
[LonelyPlanets](https://github.com/spzwart/LonelyPlanets)

Relevant papers that use LonelyPlanets:
- [Stability of Multiplanetary Systems in Star Clusters](https://arxiv.org/pdf/1706.03789)
- [The signatures of the parental cluster on field planetary
systems](https://arxiv.org/pdf/1711.01274)

Link for Example Codes:
https://github.com/amusecode/amuse/blob/ce21df1cc58297c0e3ab0d7afa3b4ed2fa4cea24/examples/simple/orbit_in_potential.py

[Tutorial for Simulating Ultra Compact Dwarf Galaxies](https://github.com/amusecode/amuse/blob/main/examples/tutorial/tutorial.pdf)

## ToDos:
- read 4th chapter of a book  + 3.2.1 Initial condition for gravitational dynamics
- do the 8th tutorial
- understand LonelyPlanets

- Find core question we want to answer: What types of planetary systems can survive tidal disruption of the parent galaxy?
                                        (At what timescale does tidal stripping occur?)

## Initial conditions 
- number of stars in claster - typically 100-10000 but smaller more popular 
- maximum and minimum mases of a stars in a claster - set for now 1 to 30
- type of model to make a cluster - new_fractal_cluster_model? 
- velocities of a cluster - 3 parameters
- position of a cluster - 3 parameters
- size of a cluster - for now 1 kpc 

- hill radius? 100-1000 au


 - 
## Info maybe useful in future 
- GalactICs (Kuijken & Dubinski, 1995), which is designed to set up a self-consistent galaxy model with a disk, bulge, and dark halo - in book 4.4.5 Merging Galaxies

## Papers 
- https://arxiv.org/abs/2207.09752
- https://ui.adsabs.harvard.edu/abs/2019PhDT........53N/abstract
- https://arxiv.org/abs/2310.03327
- https://www.aanda.org/articles/aa/full_html/2014/05/aa23124-13/aa23124-13.html
- https://academic.oup.com/mnras/article/290/3/490/996086?login=true
- https://www.aanda.org/articles/aa/full_html/2014/05/aa23124-13/aa23124-13.html
- https://academic.oup.com/mnras/article/536/1/298/7911845?login=true

## Grading
1 star, "evolution of disk over time"- plot
