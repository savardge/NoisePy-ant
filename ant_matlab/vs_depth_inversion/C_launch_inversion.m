function [] = C_launch_inversion(ind_lin_start, N_ind) 
% Script to launch depth inversion for a range of grid indices
tic 

%% Define environment variables and get number of CPUs to use

run_on_cluster = true;
if run_on_cluster
    % setenv('LD_LIBRARY_PATH', ['/home/users/s/savardg/anaconda3/envs/geopsy/lib:',getenv('LD_LIBRARY_PATH')]); % Yggdrasil
    setenv('LD_LIBRARY_PATH', ['/home/users/s/savardg/.conda/envs/geopsy/lib:',getenv('LD_LIBRARY_PATH')]); % Baobab
    num_cores = str2num(getenv('SLURM_CPUS_PER_TASK')); % number of cores for parallel computing
else
    setenv('LD_LIBRARY_PATH', ['/usr/lib/x86_64-linux-gnu:',getenv('LD_LIBRARY_PATH')]); % Laptop
    num_cores = 8;
end
disp(['Number of cores: ' num2str(num_cores)])

%% create parallel pool

p = gcp('nocreate'); % If no pool, do not create new one yet.
if isempty(p)
    parpool(num_cores); % creates parallel pool
elseif p.NumWorkers ~= num_cores
    delete(p); % deletes existing one
    parpool(num_cores); % creates parallel pool
end


%% Define paths for inputs and output

params_file = 'params_gpdc_T0.5-4.5s.mat'
datadir = '../../data-aargau/run4_dcV2_mul3_g500m/vs-model'
input_file = [datadir '/data_picked_all_data_LC0.8_sigma4_ZZ-ZR.mat']
output_folder = [datadir '/run1_dv40_dz50m_N60_14L_ZZ-ZR']

if ~exist(output_folder,'dir')
    mkdir(output_folder);
end

%% Inversion params

load(params_file, 'T_vec', 'N_samp_read', 'min_freq', 'max_freq', 'sampling_type'); % N_samp_read number of sample outputed by gpdc with current parameters. 

N_ini = 30000; % 20000 default; number of initial random models to generate
N_best = 100; % 100; number of best fitting cells to keep
N_resamp_cell = 5;  % 5; number of new samples per best cells
N_iter = 60; % 20 default, number of iterations

bool_v_increase = true; % set to true to force velocity to increase with depth; set to false otherwise
min_dz = 50; % 100 default, minimum thickness of layer in m
max_dv = 40/100; % 30/100 default, max relative velocity change between subsequent layers

%% Prior bounds
range_mat = [ ... % On each line: Depth min, depth max, vmin, vmax
         100         200   500        4000; ... 
         200         400   500        4000; ...
         400         600   500        4000; ...
         600         800   500        4000; ...
         800        1000   500        4000; ...
        1000        1500   500        4000; ...
        1500        2000  1000        4000; ...
        2000        2500  1000        4000; ...
        2500        3000  1000        4000; ...
        3000        3500  1000        4500; ...
        3500        4000  1000        4500; ...   
        4000        4500  1000        4500; ...  
        4500        5000  1000        4500; ...           
           0           0  1000        4500   ];
n_layer = size(range_mat,1);
% Depth range:
Z_range_mat = range_mat(:, 1:2);
% initial depth model as starting point for random walk (arbitrary but needs to be consistent with params)
z_start = mean(Z_range_mat,2); % take the mean of ranges as starting model

% Velocity range:
V_range_mat = range_mat(:, 3:4);
v_start = ones(n_layer, 1) .* 3.4651 *1e3;
%v_start(1:5) = 2500;
%v_start = [945 1603 2355 2355 2416 2416 2769 3115 3400 3400]';

% Sanity checks
if size(Z_range_mat,1) ~= size(V_range_mat,1)
    error('Mismatch in number of values provided for range in z and Vs')
end
if length(z_start) ~= length(v_start)
    error('Initial model has mismatch in number of values provided for z_start and v_start')
end

get_Vp = @(Vs) sqrt(3)*Vs;  % define function handle that calculates Vp in function of Vs (here using Vp/Vs ratio of sqrt(3)
get_rho = @(Vp) 0.31*Vp.^(1/4)*1000;  % define function handle that calculates rho in function of Vp (here Gardner)
% get_rho = @(Vs) 2650*ones(size(Vs));  % example of constant rho at 2650;

%% prepare picked data

% Load dispersion curves at all 2D grid indices from Vg map inversions
load(input_file,'T_pick','vg_pick_mat','raycount_total')
raycount_total = raycount_total(:);
% available variables: 'T_pick', 'vg_pick_mat','vg_pick_mat1', 'vg_pick_mat2', 'vg_pick_mat3', 'ind_x_range', 'ind_y_range', 'x_grid', 'y_grid', 'dx_grid', 'dy_grid', 'sigma', 'LC')

% Interpolate data to gpdc period values (keep only periods for which there are picks)
bool_interp = T_vec > T_pick(1) & T_vec < T_pick(end);
T_pick_interp = T_vec(bool_interp)'; % should be row vector
vg_pick_mat_interp = interp1(T_pick,vg_pick_mat,T_pick_interp);

%% Save params 

if ~exist([output_folder '/params_inv.mat'], 'file')

    % Save inversion parameters
    save([output_folder '/params_inv.mat'], ...
        'N_ini', 'N_resamp_cell', 'N_best','N_iter', ...
        'bool_v_increase','min_dz','max_dv', ...
        'Z_range_mat','V_range_mat', 'z_start','v_start', ...
        'T_pick','vg_pick_mat','T_pick_interp','vg_pick_mat_interp'); 

%     % Copy dispersion curve data
%     copyfile(input_file,[output_folder '/' input_file]);
    % Create symbolic link to dispersion curve data (save memory vs copying)
    [~,input_file_name,input_file_ext] = fileparts(input_file);
    system(['ln -s ' input_file ' ' output_folder '/' input_file_name input_file_ext])

    % Copy gpdc parameter files
    copyfile(params_file,[output_folder '/' params_file]); 

    % Copy this script
    script_name = [mfilename, '.m'];
    copyfile(script_name,[output_folder '/' script_name]);
        
end

%% loop for call inversion process

% Range of indices for which to loop over:
ind_lin_range = ind_lin_start:1:min([size(raycount_total,1), (ind_lin_start+N_ind)]);
disp(['Launching inversion for indices from ' num2str(ind_lin_range(1)) ' to ' num2str(ind_lin_range(end))])

% Iterate over each local DC curve (each (x,y) position)
parfor ind_lin = ind_lin_range

    % Check if there are rays crossing this cell
    if raycount_total(ind_lin) == 0
        continue
    end

    % Check if already processed
    output_fname = sprintf([output_folder '/output_ind_lin_%d.mat'], ind_lin);
    if exist(output_fname,'file')
        disp(['Already processed index ' num2str(ind_lin) ', skipping.'])
        continue
    end

    % Get data for this cell
    vg_pick_interp = vg_pick_mat_interp(:,ind_lin)'; % should be row vector
    
    % Call inversion routine
    [P_merge, misfit_merge, disp_mat_merge] = inversion_main(T_pick_interp,vg_pick_interp, ...
        N_ini,N_best,N_resamp_cell,N_iter, ...
        sampling_type,min_freq,max_freq,T_vec,N_samp_read, ...
        bool_v_increase,min_dz,max_dv,Z_range_mat,V_range_mat,z_start,v_start, ...
        get_rho,get_Vp,output_folder,ind_lin,run_on_cluster);
    
    %clear vg_pick_interp P_merge misfit_merge disp_mat_merge N_tot ind_sort_all ind_min vg_min zmin_glob v_min_glob Z_merge V_merge Z_min V_min %not valid on parfor
   
    % Save inversion results
    m_curr = matfile(output_fname,'writable',true); % workaround to save from within parfor
    m_curr.P_merge = P_merge;
    m_curr.misfit_merge = misfit_merge;
    m_curr.disp_mat_merge = disp_mat_merge;
        
    disp(['Inversion for ind_lin=' num2str(ind_lin) ' finished.'])
end

disp('Finished inversion for all given indices.')
toc

end
