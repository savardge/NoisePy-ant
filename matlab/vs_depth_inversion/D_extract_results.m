% Old version of script to extract of inversion results, without using slurm job arrays
% 
% Thomas planes (2019), Genevieve Savard (2023)

clear all; close all;
clc

output_fold='20220518T174950_12L_5km_nagra';


output_file = [output_fold '/result.mat']

load([output_fold '/data_picked.mat'], 'ind_x_range', 'ind_y_range', 'x_grid', 'y_grid', 'dx_grid', 'dy_grid'); 
%T_pick vg_pick_mat
%load ../../data/D_grid_and_ray_kernel/stat_grid.mat x_grid y_grid dx_grid dy_grid x_stat y_stat

load([output_fold '/params_inv.mat'], 'vg_pick_mat', 'T_pick_interp', 'V_range_mat', 'Z_range_mat');
nb_layer = size(Z_range_mat,1);
nb_profile = size(vg_pick_mat,2);
nb_T = length(T_pick_interp);

z_inf = Z_range_mat(:,1)'; z_sup = Z_range_mat(:,2)'; % to compute z and Vs from P
v_inf = V_range_mat(:,1)'; v_sup = V_range_mat(:,2)';

%% get final best fits
    
vg_lin_min = zeros(nb_profile,nb_T); % best fit
vg_lin_sol = zeros(nb_profile,nb_T); % solution built as average of best N fits

%%

% in kernel, grid nodes defined at bottom left of cell (should dblcheck); in imagesc, node is at center of cell; these new effective axes compensate for this
x_grid_eff = x_grid + dx_grid/2; 
y_grid_eff = y_grid + dy_grid/2; 
max_depth = max(Z_range_mat(:));

dz_grid = 10;
z_grid = 0:dz_grid:max_depth;
z_grid_eff = z_grid+dz_grid/2;

Vs_lin_min = zeros(nb_profile,length(z_grid));
Vs_lin_sol = zeros(nb_profile,length(z_grid));

%%

for ind_lin = 1:nb_profile

    load([output_fold '/iteration_files/output_ind_lin_' num2str(ind_lin) '.mat'], 'misfit_merge', 'P_merge', 'disp_mat_merge');
%     load([output_fold '/output_ind_lin_' num2str(ind_lin) '.mat'], 'misfit_merge', 'P_merge', 'disp_mat_merge');
        
    %% best fit
    
    [misfit_min, ind_min] = min(misfit_merge);
                
    vg_min = disp_mat_merge(ind_min,:);
    vg_lin_min(ind_lin,:) = vg_min;
    P_min = P_merge(ind_min,:);
    
    Z_min = z_inf + (z_sup - z_inf).* [P_min(1:nb_layer-1) 0];
    V_min = v_inf + (v_sup - v_inf).* P_min(nb_layer:end);
   
    %% combined solution
    
    % Keep 100 best models
    N_keep = 100;
    [misfit_merge_sorted, ind_sorted] = sort(misfit_merge,'ascend');
    bool_keep = ind_sorted(1:N_keep);
    vg_lin_sol(ind_lin,:) = mean(disp_mat_merge(bool_keep,:),1);
    
    % Keep models with misfit below threshold
%     misfit_thres = misfit_min * 1.2;% default: 1.2, threshold to find "best" models
%     bool_keep = misfit_merge < misfit_thres;
%     N_keep = sum(bool_keep);

    disp(['Keeping ', num2str(N_keep), ' out of ', num2str(length(misfit_merge)), ' best models for averaging.'])
    vg_lin_sol(ind_lin,:) = mean(disp_mat_merge(bool_keep,:),1);
    
    P_keep = P_merge(bool_keep,:);
   
    z_inf_glob = repmat(z_inf,[N_keep, 1]); z_sup_glob = repmat(z_sup,[N_keep, 1]); % to compute z and Vs from P
    v_inf_glob = repmat(v_inf,[N_keep, 1]); v_sup_glob = repmat(v_sup,[N_keep, 1]);
         
    Z_keep = z_inf_glob + (z_sup_glob - z_inf_glob).* [P_keep(:,1:nb_layer-1) zeros(N_keep,1)];
    V_keep = v_inf_glob + (v_sup_glob - v_inf_glob).* P_keep(:,nb_layer:end);
    
    %% Compute Vs_lin_min
   
    ind_top = 1;
    for layer = 1:nb_layer-1
        ind_bottom = round(Z_min(layer)/dz_grid) + 1;
        Vs_lin_min(ind_lin,ind_top:ind_bottom) = V_min(layer);
        ind_top = ind_bottom;
    end
    Vs_lin_min(ind_lin,ind_top:end) = V_min(end); %last layer
    
    %% Compute Vs_lin_sol
    
    Vs_sol_temp = zeros(length(z_grid),N_keep);
    
    for ind_keep = 1:N_keep
        ind_top = 1;
        for layer = 1:nb_layer-1
            ind_bottom = round(Z_keep(ind_keep,layer) / dz_grid) + 1;
            Vs_sol_temp(ind_top:ind_bottom,ind_keep) = V_keep(ind_keep,layer);
            ind_top = ind_bottom;
        end
        Vs_sol_temp(ind_top:end,ind_keep) = V_keep(end); %last layer
    end
    
    Vs_lin_sol(ind_lin,:) = mean(Vs_sol_temp,2)';
    
    clear P_merge misfit_merge disp_mat_merge;
    
end

Vs_min_mat = reshape(Vs_lin_min,[length(ind_x_range) length(ind_y_range) length(z_grid)]);
Vs_sol_mat = reshape(Vs_lin_sol,[length(ind_x_range) length(ind_y_range) length(z_grid)]);
vg_min_mat = reshape(vg_lin_min,[length(ind_x_range) length(ind_y_range) nb_T]);
vg_sol_mat = reshape(vg_lin_sol,[length(ind_x_range) length(ind_y_range) nb_T]);

x_profile = x_grid_eff(ind_x_range);
y_profile = y_grid_eff(ind_y_range);

disp(['Results saved in: ' output_file ])
save(output_file, 'Vs_min_mat','Vs_sol_mat','vg_min_mat','vg_sol_mat','T_pick_interp','x_profile','y_profile','z_grid_eff');




