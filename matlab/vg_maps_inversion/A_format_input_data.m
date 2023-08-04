function [] = A_format_input_data(comp)
% Create data kernels (G matrix) for a range of periods, from PICK_CELL.mat file that
% contains picked group dispersion measurements.
%
% Written by Thomas Planes (2019)
% Modified by Genevieve Savard (2023)

%% USER-DEFINED STUFF
datadir = '/media/genevieve/sandisk4TB/matlab-swant/data-aargau/run3_dcV2_mul2_g500m'
% comp = 'ZZ'

% Output path
output_path = [datadir, '/vg-maps/data_kern_' comp '/'];
if ~exist(output_path,'dir')
    mkdir(output_path);
end

% Group dispersion pick matrix file
pickfile = [datadir '/picks/all_picks_' comp '_lamb1.5_mul2.mat']

% Station info
station_file = [datadir '/dist_stat.mat']

% Period range for which to produce the data kernels
Tc_list = 0.2:0.1:6.0

% Full kernel G
kernel_file = [datadir '/grid/kernel.mat']

%% Load inputs

% Station info
load(station_file, 'DIST_mat', 'stat_list', 'net_list')
nb_stat = length(stat_list);
nb_cpl = nb_stat*(nb_stat-1)/2;
disp(['Number of stations: ' num2str(nb_stat) ', number of pairs: ' num2str(nb_cpl)])

% Load pick data
disp(['Reading pick file: ' pickfile])
load(pickfile, 'PICK_CELL')

% Load data kernel G
load(kernel_file, 'G_mat')
G_template = G_mat;
clear G_mat;

%% Loop over each period to produce data kernels
for ind_Tc=1:length(Tc_list)
    
    Tc = Tc_list(ind_Tc);
    disp(['Processing period = ', num2str(Tc)])

    G_mat = G_template; % initialize G
    TAU = zeros(nb_cpl,1); % contains group travel time (inter-station distance/group velocity)
    V_dat = zeros(nb_cpl,1); % contains group velocity (picked)
    bool_nodata = zeros(nb_cpl,1); % bool if data present or not for each station pair
    
    cpl = 0; % counter for number of measurements 
    % Iterate over each station pair and extract picked group velocity
    for ss = 1:nb_stat-1
        for rr = ss+1:nb_stat
            
            cpl = cpl+1;

            % Get station names for pair
            ssta = stat_list{ss}; % station name for virtual source
            snet = net_list{ss}; % network name for virtual source
            skey = [snet '_' ssta]; 
            rsta = stat_list{rr}; % station name for virtual receiver
            rnet = net_list{rr}; % network name for virtual receiver
            rkey = [rnet '_' rsta];

            % Extract data from PICK_CELL
            try
                tutu = getfield(PICK_CELL,skey,rkey);
            catch
                bool_nodata(cpl) = 1; % no data found
                continue
            end
            tutu = tutu';
%             tutu=PICK_CELL{ss,rr}; % this applies to previous format of
%             PICKCELL (no fieldnames)
            
            % If data found for this station pair:
            if ~isempty(tutu)
                
                T_list = tutu(:,1); V_list = tutu(:,2)/1000;  % need km
                %ind = find(T_list==Tc); %strict equality does not work
                ind = find(abs(T_list-Tc)<0.01); % Find in picked data the measurement for the given period Tc
                
                if ~isempty(ind) % If data found for this station pair and period Tc:
                    TAU(cpl) = DIST_mat(ss,rr)/V_list(ind);
                    V_dat(cpl) = V_list(ind);
%                     disp([num2str(cpl) ': ' num2str(ss) '-' num2str(rr) ' keep: ' num2str(TAU(cpl))])
                else % no data found at the given period Tc
                    bool_nodata(cpl) = 1;
%                     disp([num2str(cpl) ': ' num2str(ss) '-' num2str(rr) ' reject'])
                end
            else % no data found for this station pair
                bool_nodata(cpl) = 1;
%                 disp([num2str(cpl) ': ' num2str(ss) '-' num2str(rr) ' reject'])
            end
        end
    end
    
    % Find indices of station pairs without data and update G matrix and
    % data vectors accordingly.
    list_exclude = find(logical(bool_nodata)); 
    G_mat(list_exclude,:) = [];
    TAU(list_exclude) = [];
    V_dat(list_exclude) = [];
    
    v_moy = mean(V_dat); % mean group velocity

    % Save data kernel
    fname = [output_path 'data_and_kern_T' sprintf('%3.1f',Tc) '_' comp '.mat'];
    save(fname, 'TAU', 'G_mat', 'bool_nodata', 'v_moy', 'Tc');
    disp(['Data kernel saved to: ' fname])
    
    clear TAU G_mat bool_nodata v_moy Tc
    
end


