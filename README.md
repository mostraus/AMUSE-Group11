# AMUSE-Group11
Code and Content regarding the AMUSE Project from Group 11

## Topic: The survival of planetary systems in dwarf galaxy tidal streams
**Motivation**: Globular clusters and dwarf galaxies are tidally stripped; some stars with planets may end up in tidal streams.

**Question**: Can planets around stripped stars survive the tidal disruption and ejection into the Galactic halo?

**AMUSE use**: N-body for star–planet systems embedded in cluster/dwarf potential + Galactic tidal field.

**Deliverables**: Survival fraction of planets vs. cluster mass, stream orbit, and initial planet distribution.

**Why interesting**: Connects exoplanet survival to galactic dynamics.

## Notes
- Run stage 1 of LonelyPlanets
- Star cluster in potential of galaxy
- Build database of encounters
- Stage 2: planetary system + stars
- Bridge the 2 codes (hydro code for disk + N-body code for star)
- How is a protoplanetary disk perturbed by stellar cluster?

- check only if stars come closer than 2*RHill (check if it comes in perturbing distance)
- calc disk during encounter, then freeze until next (fast forward)
- write LonelyPlanets ourselves, save close encounters of certain stars with stopping condition
- no bridging, just hydro code (can do gravity too, just "sloppier")
- during encounter hydro, than check if disk is ok, jump to next encounter

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
- understand lonely planets

- Find core question we want to answer: What types of planetary systems can survive tidal disruption of the parent galaxy?
                                        (At what timescale does tidal stripping occur?)



## Info maybe usfull in future 
- GalactICs (Kuijken & Dubinski, 1995), which is designed to set up a selfconsistent galaxy model with a disk, bulge, and dark halo - in book 4.4.5 Merging Galaxies


## Grading:
1 star, "evolution of disk over time"- plot
