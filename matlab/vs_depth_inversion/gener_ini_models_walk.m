function P_ini = gener_ini_models_walk(p_walker,N_ini,Z_range_mat,V_range_mat,bool_v_increase,max_dv,min_dz)

% p_walker, starting point for random walk in valid model space; could be center of prior info
% N_ini, number of valid initial models to generate
% Z_range_mat n_layer*2 min and max depth for each layer
% V_range_mat n_layer*2 min and max Vs for each layer
% bool_v_increase boolean set to 'true' to force velocity to increase with depth
% max_dv maximum relative velocity change between subsequent layers
% min_dz minimum layer thickness

if size(Z_range_mat) ~= size(V_range_mat)
    error('error: matrices Z_mat and V_mat need to have same size');
    return
end

n_layer = size(Z_range_mat,1);
nb_dim = 2 * n_layer - 1; % number of free parameters (depth of last layer infinite)

P_ini = zeros(N_ini, nb_dim); % scaled free parameters (between 0 and 1)

%% generate large random model space through random walk

zmin = Z_range_mat(:,1)'; 
zmax = Z_range_mat(:,2)';
vmin = V_range_mat(:,1)'; 
vmax = V_range_mat(:,2)';

x_inf_temp=0;  % boundaries of rescaled space
x_sup_temp=1;

for ind_walk = 1:N_ini
    
    for ind_dim = 1:nb_dim
        
        %% add constraints (forward and backward!)
        
        [x_inf, x_sup] = check_constraint_voronoi(ind_dim,p_walker,x_inf_temp,x_sup_temp,zmin,zmax,vmin,vmax,n_layer,min_dz,max_dv,bool_v_increase);
        
        %% random displacement along current axis
        
        if x_sup > x_inf
            x_walker_new = x_inf + (x_sup - x_inf) * rand; % random on initial box + constraints
        else
            error('error range')
            %pause;
        end
        
        p_walker(ind_dim) = x_walker_new;
        
    end
    
    P_ini(ind_walk,:) = p_walker;
    
end

end

