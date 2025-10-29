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
- Stage 2: planetary system + 2 stars
- Just hydro code there (it has some gravity already involed)
- How is a protoplanetary disk perturbed by stellar cluster?

**Details**:
- write script for star cluster ($N=1000$?)
- brdige star cluster's evolution with tidal tail of galaxy (gravity + hydro)
- check only if stars come closer than $2~R_\text{Hill}$ (check if it comes within perturbing distance)
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



## Info maybe usfull in future 
- GalactICs (Kuijken & Dubinski, 1995), which is designed to set up a selfconsistent galaxy model with a disk, bulge, and dark halo - in book 4.4.5 Merging Galaxies


## Grading:
1 star, "evolution of disk over time"- plot
