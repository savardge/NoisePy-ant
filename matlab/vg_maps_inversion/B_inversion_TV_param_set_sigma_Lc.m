function [] = B_inversion_TV_param_set_sigma_Lc(Tc)
% Perform set of Tarantola-Valette type inversions at fixed Tc with different sigma and LC parameter ranges
% Period Tc given as argument. Input/output paths and inversion params need
% to be modified below. Used to determine the best sigma and LC inversion
% parameters.
% 
% Written by Thomas Planes (2019)
% Modified by Genevieve Savard (2023)

%% set up parpool
p = gcp('nocreate');
if isempty(p)
    if isempty(getenv('SLURM_CPUS_PER_TASK')) 
        sz = 4; % Default number of CPUs if not running with slurm (change as needed)        
    else % Get number of cpus defined in SBATCH definition (Slurm)
        sz = str2num(getenv('SLURM_CPUS_PER_TASK'));
    end
    disp(['Starting parallel pool with ' num2str(sz) ' workers.'])
    parpool("local",sz);
end

%% USER-DEFINED STUFF

% *** PATHS
datadir = '../../data-riehen'
comp = 'ZZ'
data_kernel_fname = [datadir,'/F_inv_2D_vg_slices/data_kern_' comp '/data_and_kern_T' sprintf('%3.1f',Tc) '_' comp '.mat']
kernel_fname = [datadir, '/D_grid_and_ray_kernel/kernel.mat'] % to get grid info

outputdir = [datadir,'/F_inv_2D_vg_slices']
if ~exists(outputdir, "dir")
    mkdir(outputdir)
end

% *** INVERSION PARAMS
% Relative data error in %
rel_err = 10/100; % could try other values or affect a varying number depending on pick confidence

% Spatial smoothing factor Lc: range to test
LC_vec = 0.1:0.1:2; 
nb_LC = length(LC_vec);
disp(['Number of LC values to test: ' num2str(nb_LC)])

% Trade-off factor between minimizing model misfit and data misfit sigma:
% range to test:
exp_list = 10.^(-4:1:2); % to construct sigma vector, 
coeff_list = (0.2:0.2:1)';
sigma_mat = repmat(coeff_list,[1 length(exp_list)]).*repmat(exp_list,[length(coeff_list) 1]);
sigma_vec = sigma_mat(:)';
% sigma_vec = 0.25:0.25:2.0;
nb_sigma = length(sigma_vec);
disp(['Number of sigma values to test: ' num2str(nb_sigma)])

%% Load model parametrization and data

% Load grid info (define model parametrization)
load(kernel_fname, 'dx_grid', 'dy_grid', 'X_GRID', 'Y_GRID', 'x_grid', 'y_grid', 'x_stat', 'y_stat')
L0 = sqrt(dx_grid^2+dy_grid^2); % size of model cells
N_m = length(x_grid)*length(y_grid); % number of model cells
x_cell = reshape(X_GRID,[N_m, 1]); 
y_cell = reshape(Y_GRID,[N_m, 1]);
X_CELL = repmat(x_cell,[1 N_m]); 
Y_CELL = repmat(y_cell,[1 N_m]);
DIST_CELL = sqrt((X_CELL-X_CELL').^2+(Y_CELL-Y_CELL').^2); % inter-cell distances
        
% Load data
load(data_kernel_fname, 'TAU', 'G_mat', 'v_moy');
d = double(TAU);  % d = group travel time data
G = G_mat; % G = kernel
clear TAU G_mat;
% N_d = length(d); % number of data points

%% Define priors

% Define a-priori data covariance matrix
Cd_vec = (rel_err*d).^2;
CD = diag(Cd_vec);
CD_inv = diag(1./Cd_vec);

% Define prior model
s_prior = 1/v_moy; % homogeneous slowness
m_prior = s_prior * ones(N_m,1);

% Structs to save results to
d_post_struc = cell(nb_sigma,nb_LC);
m_est_struc = cell(nb_sigma,nb_LC);

%% Run inversions for each combination of params (sigma,LC)
disp("Starting parfor loop")
tic
parfor ind_sigma=1:nb_sigma
    
    sigma = sigma_vec(ind_sigma);
    
    for ind_LC=1:nb_LC
        
        LC = LC_vec(ind_LC);
        
        % Define a-priori model covariance matrix
        CM = (sigma*L0/LC)^2*exp(-1/LC*DIST_CELL);
        CM_inv = inv(CM);
        
        %% T-V inversion
        
        m_est = m_prior + ( G' * CD_inv * G + CM_inv ) \ G' * CD_inv * ( d - G * m_prior);
        A = G' * CD_inv * G + CM_inv;
        b = G' * CD_inv * ( d - G * m_prior);
        d_post = G * m_est;
        
        %% save results        
        d_post_struc{ind_sigma,ind_LC} = d_post;
        m_est_struc{ind_sigma,ind_LC} = m_est;
        
    end
    disp([num2str(ind_sigma) ' out of ' num2str(nb_sigma) ' for sigma = ' num2str(sigma)]);
end
toc

outputfname = [outputdir '/inv_T' sprintf('%3.1f',Tc) '_full_params_set_' comp '.mat'];
save(outputfname,'Tc','sigma_vec','nb_sigma','LC_vec','nb_LC','d','d_post_struc','m_prior','m_est_struc','rel_err','L0','G');
disp(['All inversion outputs saved to: ' outputfname])

delete(gcp('nocreate'))

end % function end
