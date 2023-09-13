function [] = C_launch_inversion_slurmarray(ind_lin_start, N_ind) 

% setenv('LD_LIBRARY_PATH', ['/usr/lib/x86_64-linux-gnu:',getenv('LD_LIBRARY_PATH')]); % Laptop
% setenv('LD_LIBRARY_PATH', ['/home/users/s/savardg/.conda/envs/geopsy/lib:',getenv('LD_LIBRARY_PATH')]); %  Baobab
setenv('LD_LIBRARY_PATH', ['/home/savardg/anaconda3/envs/geopsy/lib:',getenv('LD_LIBRARY_PATH')]); %  Unige computer

tic

%% create output folder named with current date and time and copy current params and pickings
params_file = 'params_gpdc_T0.5-4.5s.mat';
datadir = '../../data-aargau/run3_dcV2_mul2_g500m';
input_dir = [datadir '/vs-model/all_data_LC0.8_sigma4_ZZ_bycell'];
output_folder = [datadir '/vs-model/run1_dv35_dz100m_N100_30L'];
if ~exist(output_folder,'dir')
    mkdir(output_folder);
end

%% Inversion params

load(params_file, 'T_vec', 'N_samp_read', 'min_freq', 'max_freq', 'sampling_type'); % N_samp_read number of sample outputed by gpdc with current parameters. 

N_ini = 20000; % 20000 default; number of initial random models to generate
N_best = 100; % 100; number of best fitting cells to keep
N_resamp_cell = 10;  % 5; number of new samples per best cells
N_iter = 100; % 20 default, number of iterations 

bool_v_increase = true; % set to true to force velocity to increase with depth; set to false otherwise
min_dz = 90; % 100 default, minimum thickness of layer in m
max_dv = 35/100; % 30/100 default, max relative velocity change between subsequent layers

%% Prior bounds

% Define layer depth ranges
% n_layer = 11;
% max_depth = 7000;
% Z_range_upper = linspace(max_depth/(n_layer-1),max_depth,n_layer-1)';
% Z_range_lower = [min_dz; Z_range_upper(1:end-1)];
% Z_range_mat = [Z_range_lower, Z_range_upper];
% Z_range_mat = [ Z_range_mat; 0 0]; % last layer depth needs to kept at zero (depth considered infinite)
% Define layer velocities ranges: on each line, min and max of Vs for the layer, nb of line = number of layer
% V_range_mat = ones(n_layer, 2);
% V_range_mat(:,1) = 1000;
% V_range_mat(:,2) = 4500;
% 
% % initial depth model as starting point for random walk (arbitrary but needs to be consistent with params)
% v_start = ones(n_layer, 1) .* 3.4651 *1e3; % Basel Dyer et al. 2008 homogeneous model Vp=5.96, Vp/Vs = 1.72 for granitic basement
% v_start(1) = 2000;

n_layer = 30;
vmin = 500; vmax = 3600;
range_mat = zeros(n_layer+1,4);
zmax = 4000;
thickness = zmax/n_layer;
z = 0;
for nl=1:n_layer

    range_mat(nl,1:2) = [z, z+thickness];
    if z < 1000
        range_mat(nl, 3:4) = [500 4000];
    else
        range_mat(nl, 3:4) = [500 4000];
    end

    z = z+thickness;
end
range_mat(1,1) = min_dz;
range_mat(end,3:4) = [1500 4000]


% range_mat = [ ... % On each line: Depth min, depth max, vmin, vmax
%           50         200   500        4000; ... 
%          200         400   500        4000; ...
%          400         600   500        4000; ...
%          600         800   500        4000; ...
%          800        1000   500        4000; ...
%         1000        1500   500        4000; ...
%         1500        2000  1000        4000; ...
%         2000        3000  1000        4000; ...
%         3000        5000  1000        4500; ...
%            0           0  1000        4500   ];
n_layer = size(range_mat,1);
Z_range_mat = range_mat(:, 1:2);
% initial depth model as starting point for random walk (arbitrary but needs to be consistent with params)
z_start = mean(Z_range_mat,2); % take the mean of ranges as starting model

V_range_mat = range_mat(:, 3:4);
v_start = ones(n_layer, 1) .* 3.4651 *1e3;
%v_start(1:5) = 2500;
% v_start = [945 1603 2355 2355 2416 2416 2769 3115 3400 3400]';

% Sanity checks
if size(Z_range_mat,1) ~= size(V_range_mat,1)
    error('Mismatch in number of values provided for range in z and Vs')
end
if length(z_start) ~= length(v_start)
    error('Initial model has mismatch in number of values provided for z_start and v_start')
end

% Density and Vp definitions
get_Vp = @(Vs) sqrt(3)*Vs;  % define function handle that calculates Vp in function of Vs (here using Vp/Vs ratio of sqrt(3)
%get_rho = @(Vp) 0.31*Vp.^(1/4)*1000;  % define function handle that calculates rho in function of Vp (here Gardner)
get_rho = @(Vs) 2650*ones(size(Vs));  % example of constant rho at 2650;

%% Save params 

save_var = false
if save_var %ind_lin_start == 1
    % copyfile('data_picked.mat',[output_folder '/data_picked.mat']);
    copyfile(params_file,[output_folder '/' params_file]);
    
    % Copy files used for inversion
    copyfile('C_launch_inversion_slurmarray.m',[output_folder '/C_launch_inversion_slurmarray.m']);
    
end

%% loop for call inversion process

ind_lin_range = ind_lin_start:1:(ind_lin_start+N_ind);
% Iterate over each local DC curve (each (x,y) position)
for ind_lin = ind_lin_range

    % Output file name
    output_fname = sprintf([output_folder '/output_ind_lin_%d.mat'], ind_lin);
    if exist(output_fname, 'file')
        disp(['Already processed index ' num2str(ind_lin) ', skipping.'])
        continue
    end
    % Read dispersion data
    input_file = [input_dir '/input_ind_lin_' sprintf('%d', ind_lin) '.mat'];
    if ~exist(input_file, "file")
        disp(['No input data for index ' num2str(ind_lin) ', skipping.'])
        continue
    end
    load(input_file, 'T_pick', 'vg_pick') %, 'ray_count')
    
    % Interpolate data
    bool_interp = T_vec > T_pick(1) & T_vec < T_pick(end);
    T_pick_interp = T_vec(bool_interp)'; % should be row vector
    vg_pick_interp = interp1(T_pick,vg_pick,T_pick_interp); % should be row vector
        
    % Save 
    %save([output_folder '/params_inv_' num2str(ind_lin) '.mat'], 'N_ini', 'N_resamp_cell', 'N_best','N_iter','bool_v_increase','min_dz','max_dv','Z_range_mat','V_range_mat','z_start','v_start','T_pick','vg_pick','T_pick_interp','vg_pick_interp'); 
    
    tic;
    [P_merge, misfit_merge, disp_mat_merge] = inversion_main(T_pick_interp,vg_pick_interp,N_ini,N_best,N_resamp_cell,N_iter,sampling_type,min_freq,max_freq,T_vec,N_samp_read,bool_v_increase,min_dz,max_dv,Z_range_mat,V_range_mat,z_start,v_start,get_rho,get_Vp,output_folder,ind_lin);
    elapsedInd = toc;
    disp(['Inversion for ind_lin=' num2str(ind_lin) ' finished in ' sprintf('%d', elapsedInd) 's.'])
    
    save(output_fname, ...
        'P_merge', 'misfit_merge', 'disp_mat_merge', ...
        'N_ini', 'N_resamp_cell', 'N_best','N_iter', ...
        'bool_v_increase', 'min_dz','max_dv', ...
        'Z_range_mat','V_range_mat', ...
        'z_start','v_start', ...
        'T_pick','vg_pick','T_pick_interp','vg_pick_interp' ...
    )
    
end

disp('Finished.')
toc


end
