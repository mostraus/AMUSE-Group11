# AMUSE-Group11
Code and Content regarding the AMUSE Project from Group 11

## Topic: The survival of protoplanetary disks in galaxy tidal streams
**Motivation**: Star clusters and galaxies are tidally stripped; some stars with protoplanetary disk may end up in tidal streams.

**Question**: Can protoplanetary disk around stripped stars survive the tidal disruption and ejection into the Galactic halo?

**AMUSE use**: N-body for star cluster and galaxy, hydro for interactions in galaxy potential and encounters.

**Deliverables**: Survival of disks, number of encounters, distribution of disk particles during encounter.

**Why interesting**: Connects disk survival to galactic dynamics.

## How to run the simulation properly:
### Making the cluster
- all relevant  functions for this can be found in the "cluster_evolution.py" file. If someone prefers the same functionality is also accessible in jupyter notebook "stopping_conditions.ipynb"
- To get simulation it is needed to set intitial conditions of a cluster in main - number of stars, velocity (vx,vy,vz), position (x,y,z) and encounter radious. And simulation conditions like time (t_end), step of gravity code if there is no collisions (dt, do not make it to big), starting time (t, the best choice 0) and limit after which we assume the stars collide (limit). As well as name (name) of a file to save the interactions
- The functions starting_conditions_cluster , run_evolution and save_file_with_interactions need to be run. This will produce the csv file named "name" which will be used for stage 2 - disc Encounter Simulation
- There is possibility but no need to use seet in "run_evolution" to ensure that the masses of stars in a cluster will not change between runs.

### Additional Functions cluster analysys
- There is a possibility to reuse the same cluster by singing the same seed while runing "run_evolution" and then saving it to csv file by function write_set_to_file. Then it can be read by function "read_set_from_file". Important if you are using the saved cluster you still NEED to run function "starting_conditions_cluster".
- You can plot the starting positions of a star in a cluster in XY axes by "plot_the_starting_cluster_position_xy".
- If you want to analyse the cluster evolution you shoud run "save_initial_state_of_cluster" and "get_internal_energy" to have more informations about starting parameters
- For the full functionalities functions "plot_the_starting_cluster_position_xy", "save_initial_state_of_cluster" and "get_internal_energy" need to be run before "run_evolution".
- The trajectory of a cluster can be plot with function "plot_trajectory_xy"
- THe animation of this trajectory "animate_trajectory_xy"
- To analyse the the history of an evolution is need to run "save_initial_state_of_cluster" before simulation ("run_evolution") and "start_analyzis_get_pandas_dataframe" after it. If you do not do so non of functions mentioned belowe will be working.
- "plot_initial_and_ending_position_color_by_interactions" shows two plots first with starting positions second with ending positions. The size of dots depend on their mass and colors indicate if they had or not had an encounter.
- "how_many_stars_have_interaction" - needs to get the list of thresholds and then print how many stars have an interaction closer than limits in a list.
- "how_many_pairs_during_time" - need to get two numbers and print how many unique pairs had an interations before that time
- "print_interesting_interactions" - print the key of stars with interaction with the biggest mass ratio between stars, and the star with the most interactions - usefull to choose the interaction for second stage
- "statiscit_what_initial_condition_set_to_have_interaction" - print statistical information about stars which have and do not have close encounter like mean, median and standard deviation of a mass, velocity and others

  
### Making the Disk Encounter Simulation
- all relevant functions for this can be found in the "SimulateDisk.py" file
- Important: the directories "Data", "PLOT", "DISK", and "ANIM" have to exist in the directory that this is run in
- to run all the simulations of a cluster file, use the "run_all_cluster_sims(filename)" function, where filename is the name of your cluster.csv file
- to make a single simulation with disks around both stars use the "run_sim_2disk(filename, StarID1, StarID2)" function, where filename is the name of your cluster.csv file and the Star IDs are to specify the encounter that should be simulated and have to be looked up in the csv file. If you do not want the first encounter between 2 stars, use the "index" parameter.

### Additional Functions
- making animation is possible with "load_and_animate_data(fp, fv)", where fp and fv are the filenames where the position and velocity data are stored, respectively.
- additional analysis plots can be made with "EffectOnBoundPlot_fromData(RunName, PlotName)" from the functions.py file, where the RunName is the first part of the Simulation run and PlotName is the prefix of the name for the resulting plots.

All the functions also have a documentation explaining there usage in further detail!

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
- https://www.nature.com/articles/s41550-024-02349-x.pdf

## Grading
1 star, "evolution of disk over time"- plot
