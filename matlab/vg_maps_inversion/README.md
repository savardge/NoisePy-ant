This directory contains the script to obtain group velocity maps using the matrix inversion approach of Tarantola-Valette (1982)

Requires the following inputs to get started:
PICK_CELL.mat: cell matrix containing the group velocity picks for each station pair
dist_stat.mat, stat_grid.mat: 2D grid definition and station data
kernel.mat: G matrix constructed for all station pairs
map_matrix.mat: map background for plotting

There are several steps:

A) Extract pick group velocity measurements, and create data kernels (the G matrix in d=Gm) for each period

B) For a given period, run a series of inversion with different regularization parameters (sigma and LC).

C) Plot and compare the results in B to determine the optimal sigma and LC to use. Can use L-curve and result inspection.

D) Run the group velocity map inversion on the data. Can use the script that runs the inversion for one input period (for use with a job array in slurm) or the script that does a parloop to run a range of periods.

E) Plot group velocity maps. 

F) Prepare the group velocity map results for input into the Vs depth inversion (get effective group dispersion curve at each grid cell from the inverted vg maps)

Codes originally written by Thomas Planes (2019) and modified by Genevieve Savard (2023)