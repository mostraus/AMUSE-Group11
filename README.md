# AMUSE-Group11
Code and Content regarding the AMUSE Project from Group 11

## Topic: The evolution of protoplanetary disks in disrupting star clusters
**Motivation**: Stars in star clusters inside the Milky Way experience close encounters. Some of these stars have a protoplanetary disk around them which is affected by the perturber.

**Question**: Can protoplanetary disk around affected stars survive the encounter depending on initial conditions?

**AMUSE use**: $N$-body for star cluster and galaxy, hydro for interactions in galaxy potential and encounters.

**Deliverables**: Survival of disks, number of encounters, distribution of disk particles during encounter.

**Why interesting**: Connects disk survival to galactic dynamics.

## How to run the simulation properly:
### Making the cluster
- All relevant  functions for this can be found in the `cluster_evolution.py` file. If someone prefers the notebook functionality, it is also accessible in the Jupyter notebook `stopping_conditions.ipynb`
- To do the simulation, it is needed to set intitial conditions of a cluster in main - number of stars, velocity (vx,vy,vz), position (x,y,z) and encounter radius. And simulation conditions like time (t_end), step of gravity code if there is no collisions (dt, do not make it to big), starting time (t, the best choice 0) and limit after which we assume the stars collide (limit). As well as name (name) of a file to save the interactions
- The functions `starting_conditions_cluster`, `run_evolution` and `save_file_with_interactions` need to be run. This will produce the csv file named `name` which will be used for stage 2 - disc Encounter Simulation
- There is possibility but no need to use seet in `run_evolution` to ensure that the masses of stars in a cluster will not change between runs.

### Additional Functions cluster analysis
- There is a possibility to reuse the same cluster by using the same seed while running `run_evolution` and then saving it to csv file with the function `write_set_to_file`. Then it can be read by function `read_set_from_file`. Important: if you are using the saved cluster you still NEED to run function `starting_conditions_cluster`.
- The galaxy potential can be change with changing the class 'MilkyWay_galaxy' or bilding the new class and MGW = 'new_class'
- You can plot the starting positions of a star in a cluster in XY axes with `plot_the_starting_cluster_position_xy`.
- If you want to analyse the cluster evolution you shoud run `save_initial_state_of_cluster` and `get_internal_energy` to have more informations about starting parameters
- For the full functionalities functions `plot_the_starting_cluster_position_xy`, `save_initial_state_of_cluster` and `get_internal_energy` need to be run before `run_evolution`.
- The trajectory of a cluster can be plot with function `plot_trajectory_xy`
- The animation of this trajectory `animate_trajectory_xy`
- To analyse the the history of an evolution, you need to run `save_initial_state_of_cluster` before the simulation (`run_evolution`) and `start_analyzis_get_pandas_dataframe` after it. If you do not do so, none of the functions mentioned below will be working.
- `plot_initial_and_ending_position_color_by_interactions` shows two plots: first with starting positions, second with ending positions. The size of dots depend on their mass and colors indicate if they had an encounter or not.
- `how_many_stars_have_interaction` - needs to get the list of thresholds and then print how many stars have an interaction closer than limits in a list.
- `how_many_pairs_during_time` - need to get two numbers and print how many unique pairs had an interations before that time
- `print_interesting_interactions` - print the key of stars with interaction with the biggest mass ratio between stars, and the star with the most interactions - useful to choose the interaction for second stage
- `statiscit_what_initial_condition_set_to_have_interaction` - print statistical information about stars which have and do not have close encounter like mean, median and standard deviation of a mass, velocity and others

  
### Making the Disk Encounter Simulation
- All relevant functions for this can be found in the `SimulateDisk.py` file
- Important: the directories `Data`, `PLOT`, `DISK`, and `ANIM` have to exist in the directory that this is run in
- To run all the simulations of a cluster file, use the `run_all_cluster_sims(filename)` function, where filename is the name of your `cluster.csv` file
- To make a single simulation with disks around both stars use the `run_sim_2disk(filename, StarID1, StarID2)` function, where filename is the name of your `cluster.csv` file and the Star IDs are to specify the encounter that should be simulated and have to be looked up in the csv file. If you do not want the first encounter between 2 stars, use the `index` parameter.

### Additional Functions
- Making animation is possible with `load_and_animate_data(fp, fv)`, where `fp` and `fv` are the filenames where the position and velocity data are stored, respectively.
- Additional analysis plots can be made with `EffectOnBoundPlot_fromData(RunName, PlotName)` from the `functions.py` file, where the `RunName` is the first part of the Simulation run and `PlotName` is the prefix of the name for the resulting plots.

All the functions also have a documentation explaining there usage in further detail!
