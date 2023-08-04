function [x_inf, x_sup]  =  check_constraint_voronoi(ind_dim,p_walker,x_inf_temp,x_sup_temp,zmin,zmax,vmin,vmax,n_layer,min_dz,max_dv,bool_v_increase)

% possibility of v decrease corrected on Nov. 27th 2018; to verify!

%% add constraints (forward and backward!)
        
        if ind_dim == 1 % first depth layer
        
            z_below = zmin(ind_dim+1)+p_walker(ind_dim+1)*(zmax(ind_dim+1)-zmin(ind_dim+1)); %depth layer below
            z_max_cond = z_below-min_dz; % maximum z for current layer for constraint
            p_max_cond = (z_max_cond-zmin(ind_dim))/(zmax(ind_dim)-zmin(ind_dim));
            x_inf = x_inf_temp; 
            x_sup = min([x_sup_temp p_max_cond]);  
            %clear z_below z_max_cond p_max_cond
            
        elseif ind_dim > 1 && ind_dim < n_layer-1 % depth layers except first one and last one 
            
            z_above = zmin(ind_dim-1)+p_walker(ind_dim-1)*(zmax(ind_dim-1)-zmin(ind_dim-1)); %depth layer above
            z_below = zmin(ind_dim+1)+p_walker(ind_dim+1)*(zmax(ind_dim+1)-zmin(ind_dim+1)); %depth layer below
            z_min_cond = z_above+min_dz; % minimum z for current layer for constraint
            z_max_cond = z_below-min_dz; % maximum z for current layer for constraint
            p_min_cond = (z_min_cond-zmin(ind_dim))/(zmax(ind_dim)-zmin(ind_dim));
            p_max_cond = (z_max_cond-zmin(ind_dim))/(zmax(ind_dim)-zmin(ind_dim));
            x_inf = max([x_inf_temp p_min_cond]); % depth needs to increase by at least min_dz_meter
            x_sup = min([x_sup_temp p_max_cond]); % 
            %clear z_above z_min_cond p_min_cond z_below z_max_cond p_max_cond
           
        elseif ind_dim == n_layer-1 % last depth layer (except the infinite)
            
            z_above = zmin(ind_dim-1)+p_walker(ind_dim-1)*(zmax(ind_dim-1)-zmin(ind_dim-1)); %depth layer above
            z_min_cond = z_above+min_dz; % minimum z for current layer for constraint
            p_min_cond = (z_min_cond-zmin(ind_dim))/(zmax(ind_dim)-zmin(ind_dim));
            x_inf = max([x_inf_temp p_min_cond]); % depth needs to increase by at least min_dz_meter
            x_sup = x_sup_temp; % 
            %clear z_above z_min_cond p_min_cond        
       
        elseif ind_dim == n_layer % first velocity layer
            
            ind_v_curr = ind_dim-n_layer+1;
            v_below = vmin(ind_v_curr+1)+p_walker(ind_dim+1)*(vmax(ind_v_curr+1)-vmin(ind_v_curr+1)); %Vs layer above
            
            if bool_v_increase
                v_max_cond = v_below;
            else
                %v_max_cond = v_below*(1+max_dv);
                v_max_cond = v_below/(1-max_dv); %so that dv considered from given layer to layer below
            end
                
            v_min_cond = v_below/(1+max_dv);
            p_min_cond = (v_min_cond-vmin(ind_v_curr))/(vmax(ind_v_curr)-vmin(ind_v_curr));
            p_max_cond = (v_max_cond-vmin(ind_v_curr))/(vmax(ind_v_curr)-vmin(ind_v_curr));
            x_inf = max([x_inf_temp p_min_cond]); % 
            x_sup = min([x_sup_temp p_max_cond]); % 
            %clear v_below v_min_cond v_max_cond p_min_cond p_max_cond
            
        elseif ind_dim > n_layer && ind_dim < 2*n_layer-1 % velocity layers except first one and last one
            
            ind_v_curr = ind_dim-n_layer+1;
            
            v_above = vmin(ind_v_curr-1)+p_walker(ind_dim-1)*(vmax(ind_v_curr-1)-vmin(ind_v_curr-1)); %Vs layer above
            v_below = vmin(ind_v_curr+1)+p_walker(ind_dim+1)*(vmax(ind_v_curr+1)-vmin(ind_v_curr+1)); %Vs layer above
            
            if bool_v_increase
                v_min_cond1 = v_above; % velocity needs to increase (should add condition)
                v_max_cond1 = v_below;
            else
                v_min_cond1 = v_above*(1-max_dv);
                v_max_cond1 = v_below/(1-max_dv);
            end
            
            v_max_cond2 = v_above*(1+max_dv);
            v_min_cond2 = v_below/(1+max_dv); 

            v_min_cond = max([v_min_cond1 v_min_cond2]);
            v_max_cond = min([v_max_cond1 v_max_cond2]);            
            p_min_cond = (v_min_cond-vmin(ind_v_curr))/(vmax(ind_v_curr)-vmin(ind_v_curr));
            p_max_cond = (v_max_cond-vmin(ind_v_curr))/(vmax(ind_v_curr)-vmin(ind_v_curr));
            x_inf = max([x_inf_temp p_min_cond]); % velocity needs to increase
            x_sup = min([x_sup_temp p_max_cond]); % velocity is maxxed
            %clear v_above v_min_cond1 v_min_cond2 v_min_cond v_below v_max_cond1 v_max_cond2 v_max_cond p_min_cond p_max_cond
            
        else % last velocity layer
          
            ind_v_curr = ind_dim-n_layer+1;
            v_above = vmin(ind_v_curr-1)+p_walker(ind_dim-1)*(vmax(ind_v_curr-1)-vmin(ind_v_curr-1)); %Vs layer above
            
            if bool_v_increase
                v_min_cond = v_above; %
            else
                v_min_cond = v_above*(1-max_dv); %
            end
            
            v_max_cond = v_above*(1+max_dv);
            p_min_cond = (v_min_cond-vmin(ind_v_curr))/(vmax(ind_v_curr)-vmin(ind_v_curr));
            p_max_cond = (v_max_cond-vmin(ind_v_curr))/(vmax(ind_v_curr)-vmin(ind_v_curr));
            x_inf = max([x_inf_temp p_min_cond]); %
            x_sup = min([x_sup_temp p_max_cond]); %
            %clear v_above v_min_cond v_max_cond p_min_cond p_max_cond
           
        end
        
        %clear x_inf_temp x_sup_temp
        
end
