function [P_merge, misfit_merge, disp_mat_merge] = inversion_main(T_pick,vg_pick,N_ini,N_best,N_resamp_cell,N_iter,sampling_type,min_freq,max_freq,T_vec,N_samp_read,bool_v_increase,min_dz,max_dv,Z_range_mat,V_range_mat,z_start,v_start,get_rho,get_Vp,output_fold,ind_lin,run_on_cluster)
% Main inversion script to get Vs from given dispersion curve, called by C_launch_inversion*.m
% MUST MODIFY PATH TO gpdc BELOW to fit your computer setup
% 
% Thomas Planes (2019), Genevieve Savard (2023)

verbose = true;
if run_on_cluster
	gpdc_exec = '/home/share/cdff/geopsypack-core/bin/gpdc'; % Yggdrasil
%     gpdc_exec = '/home/users/s/savardg/Geopsy/bin/gpdc'; % Baobab
    verbose = false;
else
    gpdc_exec = '/usr/local/Geopsy.org/bin/gpdc';    
end

%% get starting point in normalized units

pz = (z_start(1:end-1) - Z_range_mat(1:end-1,1))./ (Z_range_mat(1:end-1,2) - Z_range_mat(1:end-1,1));
pv = (v_start - V_range_mat(:,1))./ (V_range_mat(:,2) - V_range_mat(:,1));
p_walker = [pz' pv'];  % starting point of randow walk in normalized units
clear z_start v_start pz pv;

%% generate large random model space

if verbose; disp(['Generating ' num2str(N_ini) ' initial random models...']); end
P_ini = gener_ini_models_walk(p_walker,N_ini,Z_range_mat,V_range_mat,bool_v_increase,max_dv,min_dz);
if verbose; disp('done generating initial random models.'); end

zmin = repmat(Z_range_mat(:,1)',[N_ini, 1]); zmax = repmat(Z_range_mat(:,2)',[N_ini, 1]);
vmin = repmat(V_range_mat(:,1)',[N_ini, 1]); vmax = repmat(V_range_mat(:,2)',[N_ini, 1]);

nb_layer = size(Z_range_mat,1);
Z_ini = zmin + (zmax - zmin).* [P_ini(:,1:nb_layer-1) zeros(N_ini,1)];
V_ini = vmin + (vmax-vmin).* P_ini(:,nb_layer:end);

%save(['./' output_fold '/init_models.mat'], 'Z_range_mat', 'V_range_mat', 'P_ini', 'Z_ini', 'V_ini', 'N_ini', 'nb_layer', 'min_dz', 'max_dv', 'bool_v_increase'); % save initial models

%% make initial models file

if verbose; disp('Writing initial models file...'); end
if run_on_cluster
    mod_name = ['/scratch/init_models_' num2str(nb_layer) 'L_ind_lin' num2str(ind_lin)];
else
    mod_name = [output_fold '/init_models_' num2str(nb_layer) 'L_ind_lin' num2str(ind_lin)];
end

Vp_ini = get_Vp(V_ini);
rho_ini = get_rho(Vp_ini);

bool_fail = make_model_file(mod_name,V_ini,Vp_ini,rho_ini,Z_ini); % should change name for different points?

if bool_fail ~= 0
    error('Error in writing initial models file')    
end

if verbose; disp('done making initial models file.'); end

%% run routine to get all initial disp curves

if verbose; disp('Computing initial dispersion curves with gpdc...'); end
fname_in = [mod_name '.model']; % should change name for different points?
fname_out = [mod_name '.disp'];
%exec_line = ['!' gpdc_exec ' -group -f -s ' sampling_type ' -min ' num2str(min_freq) ' -max ' num2str(max_freq) ' ' fname_in '  > ' fname_out]; % -f added so doesnt stop when one curve cannot be computed
%eval(exec_line); %better keep in ant_matlab prompt so that release command only when finished
exec_line = [gpdc_exec ' -group -f -s ' sampling_type ' -min ' num2str(min_freq) ' -max ' num2str(max_freq) ' ' fname_in '  > ' fname_out]; % -f added so doesnt stop when one curve cannot be computed
[status, cmdout] = system(exec_line);
if status ~= 0
    error(['Failed to execute gpdc: ', cmdout])
end    
if verbose; disp('done with calculation of dispersion curves for initial models.'); end
delete(fname_in); % delete model file once read

%% read disp data
% could use textscan instead of custom function?

if verbose; disp(['Reading output file ' mod_name '...']); end
disp_mat = read_disp_data(mod_name,N_ini,N_samp_read); 
%save(['./' output_fold '/init_disp.mat'], 'disp_mat', 'freq_vec', 'N_ini');
if verbose; disp('done reading output file.'); end
delete(fname_out); % delete disp file once read

%% compute initial misfit

bool_T_keep = T_vec >= T_pick(end) & T_vec <= T_pick(1);
disp_mat_ini = disp_mat(:,bool_T_keep); clear disp_mat T_vec;
vg_pick_glob = repmat(vg_pick,[size(disp_mat_ini,1) 1]);
misfit_ini = rms((disp_mat_ini - vg_pick_glob)./ vg_pick_glob, 2);
[~, ind_sort] = sort(misfit_ini, 1, 'descend');
ind_best = ind_sort(end-N_best+1:end);
clear vg_pick_glob;

%% iterate; resample best cells

P_merge = P_ini; % will contain all data
misfit_merge = misfit_ini;
disp_mat_merge = disp_mat_ini;

zmin = Z_range_mat(:,1)'; 
zmax = Z_range_mat(:,2)'; % to compute z and Vs from P
vmin = V_range_mat(:,1)'; 
vmax = V_range_mat(:,2)';

N_resamp = N_best * N_resamp_cell;
N_free = 2 * nb_layer - 1; % number of free params (last layer depth infinite)

%% Monte Carlo loop
if verbose; disp(['Starting iteration procedure, resampling ' num2str(N_resamp_cell) ' models in each of the ' num2str(N_best) ' best cells:']); end
for ind_iter = 1:N_iter
    
    disp(['Starting iteration ' num2str(ind_iter) '/' num2str(N_iter) '...'])
    
    %P_best=P_merge(ind_best,:);
    %P_best_min=min(P_best,[],1);
    %P_best_max=max(P_best,[],1);
    %delta_P=P_best_max-P_best_min; % used to scale distances within voronoi resampling
    
    P_resamp_glob = zeros(N_resamp_cell,N_best,N_free);
    
    for ind_resamp = 1:N_best
        
        ind_best_curr = ind_best(ind_resamp);        
        P_curr = resamp_voronoi_constrained(P_merge,ind_best_curr,N_resamp_cell,max_dv,min_dz,Z_range_mat,V_range_mat,bool_v_increase); %current cell scaled coordinates (between 0 and 1)
        P_resamp_glob(:,ind_resamp,:) = P_curr;  
        
    end
    
    P_resamp = reshape(P_resamp_glob,[N_resamp, N_free]);
    
    bool_zeros = sum(P_resamp,2)==0; % should check if this can removed now??
    
    P_resamp(bool_zeros,:) = []; % remove non resamped ones
    N_resamp_eff = size(P_resamp, 1); % number of effectively resampled models
    
    zmin_vec = repmat(zmin,[N_resamp_eff, 1]); 
    zmax_vec = repmat(zmax,[N_resamp_eff, 1]); % to compute z and Vs from P
    vmin_vec = repmat(vmin,[N_resamp_eff, 1]); 
    vmax_vec = repmat(vmax,[N_resamp_eff, 1]);
    
    Z_resamp = zmin_vec + (zmax_vec - zmin_vec).* [P_resamp(:,1:nb_layer-1) zeros(N_resamp_eff,1)];
    V_resamp = vmin_vec + (vmax_vec - vmin_vec).* P_resamp(:,nb_layer:end);
    
    
    %% make models file
    
    Vs_set = V_resamp;
    Vp_set = get_Vp(Vs_set);
    rho_set = get_rho(Vp_set);
    Z_set = Z_resamp;
    
    if run_on_cluster
        mod_name = ['/scratch/iter_' num2str(ind_iter) '_ind_lin' num2str(ind_lin)];
    else
        mod_name = [output_fold '/iter_' num2str(ind_iter) '_ind_lin' num2str(ind_lin)];
    end
    
    bool_fail = make_model_file(mod_name,Vs_set,Vp_set,rho_set,Z_set);
    if bool_fail ~= 0
        warning(['Error in writing ' mod_name ' model file'])        
    end
    
    clear Vs_set Vp_set rho_set Z_set;
    
    %% compute and read new disp curves    
    fname_model = [mod_name '.model'];
    fname_disp = [mod_name '.disp'];
    % exec_line = ['!' gpdc_exec ' -group -f -s ' sampling_type ' -min ' num2str(min_freq) ' -max ' num2str(max_freq) ' ' fname_model '  > ' fname_disp]; % -f added so doesnt stop when one curve cannot be computed
    %eval(exec_line);
    exec_line = [gpdc_exec ' -group -f -s ' sampling_type ' -min ' num2str(min_freq) ' -max ' num2str(max_freq) ' ' fname_model '  > ' fname_disp]; % -f added so doesnt stop when one curve cannot be computed
    [status, cmdout] = system(exec_line);
    if status ~= 0
        error(['Failed to execute gpdc: ', cmdout])
    end    
    
%     pause(0.1); % GS uncommented
    disp_resamp_raw = read_disp_data(mod_name, N_resamp_eff, N_samp_read);
    %save(['disp_iter_' num2str(ind_iter) '.mat'],'disp_resamp','Z_resamp','V_resamp','P_resamp');
    delete(fname_model); % delete model file once read
    delete(fname_disp); % delete disp file once read
    
    %% compute new misfit
    
    disp_resamp = disp_resamp_raw(:,bool_T_keep); clear disp_resamp_raw; 
    vg_pick_glob = repmat(vg_pick, [size(disp_resamp, 1) 1]);
    misfit_resamp = rms((disp_resamp - vg_pick_glob)./ vg_pick_glob, 2);
    clear vg_pick_glob;    
    
    %% Misfit info
    if verbose; fprintf('Min misfit: %f, difference from initial models: %f percent\n',min(misfit_resamp),(min(misfit_resamp)-min(misfit_ini))/min(misfit_ini)*100); end

    %% merge results to previous ones

    P_merge = [P_merge; P_resamp];
    misfit_merge = [misfit_merge; misfit_resamp];
    disp_mat_merge = [disp_mat_merge; disp_resamp];
    
    [~, ind_sort] = sort(misfit_merge,1,'descend');
    ind_best = ind_sort(end-N_best+1:end);

    if verbose; disp(['Iteration ' num2str(ind_iter) '/' num2str(N_iter) ' done']); end

end

if verbose; disp('Inversion process completed!'); end

end
