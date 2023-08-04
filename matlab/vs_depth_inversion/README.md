These scripts perform the depth inversion step, to go from a collection of Vg maps to a 3D Vs model, by inverting at each 2D cell position the effective dispersion curve, derived from vg maps.

The steps are:

A) Extract the Vg maps produced previously, and calculate the local dispersion curves at each 2D grid cell position.

B) Do a dummy gpdc run with the desired min-max period bounds corresponding to the Vg maps used, and get the output format from gpdc (needed to read correctly gpdc output in C)

C) Launch Vs inversion in parallel, the Vs inversions for each grid cell are independents from each other. Use the C_Launch_inversion_slurmarray.m to leverage the full potential of the HPC cluster (run multiple jobs for different index ranges). Otherwise, the C_launch_inversion.m uses parfor loop, which means only the max number of CPUs on ONE node can be used (vs being able to use multiple nodes)

D) Extract the inversion results and merged them into an output file in a format practical for plotting.

E) Plot the results, customize the script to your needs. See example scripts for inspiration. 

