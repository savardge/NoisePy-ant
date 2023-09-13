% Extract depth inversion result for 1 cell and plot distribution of
% proposed models
% 
% Genevieve Savard (2023)

%% USER INPUTS
clear all; %close all
datadir = '../../data-aargau/run3_dcV2_mul2_g500m'
% output_folder = [datadir '/vs-model/run0_dv60_dz50m_N100_10L'];
output_folder = [datadir '/vs-model/run1_dv35_dz100m_N100_14L_ZZ'];
ind_lin = 1673 %1673; % run3: Riniken 1673, Bottsetin 2449, Leuggern 2720

%% Get model 
if exist([output_folder '/params_inv_' num2str(ind_lin) '.mat'],'file')
    load([output_folder '/params_inv_' num2str(ind_lin) '.mat'], 'N_ini', 'N_resamp_cell', 'N_best','N_iter','bool_v_increase','min_dz','max_dv','Z_range_mat','V_range_mat','z_start','v_start','T_pick','vg_pick','T_pick_interp','vg_pick_interp'); 
end
load([output_folder '/output_ind_lin_' num2str(ind_lin) '.mat'])

nb_layer = size(Z_range_mat,1);
nb_T = length(T_pick_interp);

% Prior boundaries
z_inf = Z_range_mat(:,1)'; z_sup = Z_range_mat(:,2)'; % to compute z and Vs from P
v_inf = V_range_mat(:,1)'; v_sup = V_range_mat(:,2)';

% Depth grid
max_depth = max(Z_range_mat(:,2));
dz_grid = 10; % interpolation spacing along depth
z_grid = 0:dz_grid:max_depth;
z_grid_eff = z_grid+dz_grid/2;

%% best fit (min misfit)

[misfit_min, ind_min] = min(misfit_merge);

vg_min = disp_mat_merge(ind_min,:);
vg_lin_min(ind_lin,:) = vg_min;
P_min = P_merge(ind_min,:);

Z_min = z_inf + (z_sup - z_inf).* [P_min(1:nb_layer-1) 0];
V_min = v_inf + (v_sup - v_inf).* P_min(nb_layer:end);

%% combined solution (average best N_keep models)

% Option 1: keep all models within some misfit threshold
% misfit_thres = misfit_min * 1.1;% default: 1.2, threshold to find "best" models
% bool_keep = misfit_merge < misfit_thres;
% N_keep = sum(bool_keep);
% vg_lin_sol(ind_lin,:) = mean(disp_mat_merge(bool_keep,:),1);

% Option 2: Keep N_keep best models
N_keep = 100;
[misfit_merge_sorted, ind_sorted] = sort(misfit_merge,'ascend');
bool_keep = ind_sorted(1:N_keep);
vg_lin_sol(ind_lin,:) = mean(disp_mat_merge(bool_keep,:),1);

disp(['Keeping ', num2str(N_keep), ' out of ', num2str(length(misfit_merge)), ' best models for averaging.'])
P_keep = P_merge(bool_keep,:);

z_inf_glob = repmat(z_inf,[N_keep, 1]);
z_sup_glob = repmat(z_sup,[N_keep, 1]); % to compute z and Vs from P
v_inf_glob = repmat(v_inf,[N_keep, 1]);
v_sup_glob = repmat(v_sup,[N_keep, 1]);

Z_keep = z_inf_glob + (z_sup_glob - z_inf_glob).* [P_keep(:,1:nb_layer-1) zeros(N_keep,1)];
V_keep = v_inf_glob + (v_sup_glob - v_inf_glob).* P_keep(:,nb_layer:end);

% Initialize
nb_profile = 1;
Vs_lin_min = zeros(nb_profile,length(z_grid));
Vs_lin_sol = zeros(nb_profile,length(z_grid));

% Compute Vs_lin_min
ind_top = 1;
for layer = 1:nb_layer-1
    ind_bottom = round(Z_min(layer)/dz_grid) + 1;
    Vs_lin_min(ind_top:ind_bottom) = V_min(layer);
    ind_top = ind_bottom;
end
Vs_lin_min(ind_top:end) = V_min(end); %last layer

% Compute Vs_lin_sol

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

Vs_lin_sol = mean(Vs_sol_temp,2)';


%% Distribution for Vs and depth  of each layer
for layer = 1:nb_layer-1
    Z_all = z_inf(layer) + (z_sup(layer) - z_inf(layer)).* P_merge(:,layer);
    V_all = v_inf(layer) + (v_sup(layer) - v_inf(layer)).* P_merge(:,nb_layer+layer-1);
    figure(1); clf
    h1=subplot(2,2,1); hist(Z_all,100); title(['Depth layer ' num2str(layer)])
    h3=subplot(2,2,3); hist(Z_keep(:,layer),100); title('Best')
    h2=subplot(2,2,2); hist(V_all,100); title(['Vs layer ' num2str(layer)])
    h4=subplot(2,2,4); hist(V_keep(:,layer),100); title('Best')
    linkaxes([h1,h3], 'x')
    linkaxes([h2,h4], 'x')
    pause(0.5)
end

%% Plot dispersion curve distribution and fit
vbins = min(V_range_mat(:)):0.1:max(V_range_mat(:));
tbins = min(T_pick_interp):0.1:max(T_pick_interp);
T_pick_interp_merge = repmat(T_pick_interp,[size(disp_mat_merge,1), 1]);

% figure(2); clf;
figure; clf;
ax1=subplot(1,2,1); cla;
h = histogram2(T_pick_interp_merge(:),disp_mat_merge(:), 'XBinEdges',tbins,'YBinEdges',vbins, 'DisplayStyle','tile','ShowEmptyBins','off');
hold on; plot(T_pick, vg_pick, 'k-o', 'LineWidth',2, 'DisplayName','Data'); 
xlabel('Period [s]'); ylabel('Group velocity [m/s]'); grid on; title('Distribution of all proposed curves')
ax2=subplot(1,2,2);cla;
plot(T_pick_interp, disp_mat_merge(bool_keep,:)'); title('100 best curves')
hold on; plot(T_pick, vg_pick, 'k-o', 'LineWidth',2, 'DisplayName','Data'); 
linkaxes([ax1,ax2],'xy')
xlabel('Period [s]'); ylabel('Group velocity [m/s]'); grid on
% ylim([1000,3000])
xlim([min(tbins),max(tbins)]); %ylim([min(vbins),max(vbins)])


%% Plot Vs model distribution

% Get layer cake models
Nmodels = size(P_merge,1);
Z_cake_all = zeros(Nmodels, 2*nb_layer -3);
V_cake_all = zeros(Nmodels, 2*nb_layer -3);
ilay = 0;
for layer = 1:nb_layer-1
    % Depth and velocity top
    ilay = ilay + 1;
    Z_cake_all(:,ilay) = z_inf(layer) + (z_sup(layer) - z_inf(layer)).* P_merge(:,layer);    
    V_cake_all(:,ilay) = v_inf(layer) + (v_sup(layer) - v_inf(layer)).* P_merge(:,nb_layer+layer-1);
    % Depth and velocity bottom
    ilay = ilay + 1;
    Z_cake_all(:,ilay) = z_inf(layer+1) + (z_sup(layer+1) - z_inf(layer+1)).* P_merge(:,layer+1);    
    V_cake_all(:,ilay) = v_inf(layer) + (v_sup(layer) - v_inf(layer)).* P_merge(:,nb_layer+layer-1);
    
    disp(['Model 1: Layer #' num2str(layer) ': Z top-bottom: ' num2str(Z_cake_all(1,ilay-1)) '-' num2str(Z_cake_all(1,ilay)) ' with Vs = ' num2str(V_cake_all(1,ilay))])
end
% Change 0 for half space to max Z in prior range
Zmax = max(Z_range_mat(:,2));
Z_cake_all(:,end) = Zmax;
% Add point for surface
Z_cake_all = [zeros(Nmodels,1) , Z_cake_all];
V_cake_all = [V_cake_all(:,1) , V_cake_all];

% figure(3); 
figure; 
clf
histogram2(V_cake_all(:),-Z_cake_all(:),[200 200],'DisplayStyle','tile','ShowEmptyBins','off','LineStyle','none');
hold on
plot(V_cake_all(ind_min,:),-Z_cake_all(ind_min,:),'k-','LineWidth',2)
xlabel("Vs [m/s]"); ylabel("Depth [m]"); title('Distribution of models')
plot(Vs_lin_sol, -z_grid, "r-",'LineWidth',2)
plot(Vs_lin_min, -z_grid, "k-",'LineWidth',2)
ylim([-Zmax,0])

%% Plot misfit
misfit_merge_min = zeros(N_iter+1,2);
misfit_merge_min(1,:) = [0 min(misfit_merge(1:N_ini))];
for k=1:N_iter
    n = N_ini + (k-1)*N_keep*N_resamp_cell + 1;
    misfit_merge_min(k+1,:) = [k min(misfit_merge(n:n+N_keep*N_resamp_cell-1))];
end
% figure(4); clf
figure; clf
h1=subplot(2,1,1);
plot(misfit_merge); title('Misfit for all models')
ylabel('Misfit'); xlabel('Model number')
% hold on; plot(misfit_merge_min(:,1), misfit_merge_min(:,2), 'r*', 'MarkerSize',8)
h2=subplot(2,1,2);
plot(misfit_merge_min(:,1), misfit_merge_min(:,2), 'r*', 'MarkerSize',8)
title('Minimum misfit for current iteration'); ylabel('Misfit'); xlabel('Iteration #')
% linkaxes([h1 h2], 'x')

