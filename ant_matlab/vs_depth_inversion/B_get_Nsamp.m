function [] = B_get_Nsamp() 
% Script to run gpdc on a dummy velocity model to get the exact output
% format of gpdc for the diven min/max periods and sampling type. The
% arbitrary velocity model is not important.
%
% see end of script for gpdc manual
% 
% Thomas Planes (2019), Genevieve Savard (2023)

%% Define path to geopsy conda environment libraries and executable path
%setenv('LD_LIBRARY_PATH', ['/usr/lib/x86_64-linux-gnu:',getenv('LD_LIBRARY_PATH')]); % Laptop
setenv('LD_LIBRARY_PATH', ['/home/users/s/savardg/anaconda3/envs/geopsy/lib:',getenv('LD_LIBRARY_PATH')]); % Yggdrasil

gpdc_exec = '/home/share/cdff/geopsypack-core/bin/gpdc'; % Yggdrasil

%% define an arbitrary model to get gdpc output parameters

sampling_type = 'log';
min_T = 0.5;
max_T = 4.5;
min_freq = 1/max_T; % T = 5
max_freq = 1/min_T; % T = 0.8

mod_name = ['arbitrary_T' sprintf('%3.1f', min_T) '-' sprintf('%3.1f', max_T) 's']; % model_name

% Define model (not important for params)
z_layer_bottom = [500 2000 0]; % take the mean of ranges as starting model
Vs = [2000 3000 3500];
Vp = Vs*sqrt(3);
rho = 0.31*Vp.^(1/4)*1000; % gardner's relationship plus conversion to kg/m3 for gdpc 

if length(Vs) ~= length(z_layer_bottom)
    error('Number of values in Vs and/or z_layer_bottom don''t match n_layer')
end

%% make model file

bool_fail = make_model_file(mod_name,Vs,Vp,rho,z_layer_bottom);

if bool_fail~=0
    error('error in writing model file')
    return;
end

%% run routine to get synthetic disp curve and read it

fname_in = [mod_name '.model']; % model file (input)
fname_out = [mod_name '.disp']; % dispersion file (output)
% eval(['!/Applications/Geopsy.org/3.3/gpdc -group -s ' sampling_type ' -min ' num2str(min_freq) ' -max ' num2str(max_freq) ' ' fname_in '  > ' fname_out]); % execute within ant_matlab prompt to make sure it is finished before releasing handle
eval(['!' gpdc_exec ' -group -s ' sampling_type ' -min ' num2str(min_freq) ' -max ' num2str(max_freq) ' ' fname_in '  > ' fname_out]); % execute within ant_matlab prompt to make sure it is finished before releasing handle
pause(0.1);

fid = fopen(fname_out,'rt');
C_read = textscan(fid,'%f %f','commentstyle','#');
fclose(fid);
freq_vec = C_read{1}; vg_vec = 1./C_read{2}; clear fid C_read;
T_vec = 1./freq_vec; 
N_samp_read = length(T_vec); 

figure(1);
plot(T_vec,vg_vec);
title('Output of gpdc')
xlabel('Period [s]')
ylabel('Velocity [m/s]')

fname = ['params_gpdc_T' sprintf('%3.1f', min_T) '-' sprintf('%3.1f', max_T) 's.mat']
save(fname, 'T_vec', 'N_samp_read', 'min_freq', 'max_freq', 'sampling_type')

%% from geopsy wiki
% can also concatenate models
% full help at !C:\"Program Files"\Geopsy.org\bin\gpdc -h all

% -group                    Switches to group slowness (default=phase)
% -s <sampling>             Defines the sampling type:
%                               period     regular sampling in period
%                               frequency  regular sampling in frequency
%                               log        regular sampling in log(frequency)
%                                          (default)
%   -min <min>                Minimum of range for dispersion curve (default=0.2)
%   -max <max>                Maximum of range for dispersion curve (default=20)
%   -vn <count>               Number of velocity samples (only for -grid,
%                             default=100)
%   -vmin <min>               Minimum of range for velocity (only for -grid,
%                             default=100)
%   -vmax <max>               Maximum of range for velocity (only for -grid,
%                             default=3000)
% 
% Usage: gpdc [OPTIONS] [FILES]
% 
%   Compute dispersion curves for layered models given through stdin or FILES.
%   
%   Format for layered models:
%     Line 1    <number of layers including half-space for first model>
%     Line 2    <thickness (m)> <Vp (m/s)> <Vs (m/s)> <Density (kg/m3)>[ <Qp> <Qs>]
%     ....
%     Line n    0 <Vp (m/s)> <Vs (m/s)> <Density (kg/m3)>[ <Qp> <Qs>]
%     Line n+1  <number of layers including half-space for second model>
%     ....
%   
%   Quality factors are not mandatory. If not specified, pure elastic computation is performed. Any number of models can be given as input.
% 
% Gpdc options: [level=0]
%   -step <s>                 Step between frequencies (default=1.025 for log scale
%   -n <count>                Number of samples (default=use step to get number of samples)
%   -L <n modes>              Number of Love modes (default=0)
%   -R <n modes>              Number of Rayleigh modes (default=1)
%   -one-mode                 Instead of outputting all modes (see options '-R' and '-L'), output only the highest one.
%   -f                        Does not return immediately if dispersion curve cannot be calculated.
%   -grid <L | R>             Ouput a grid of the wave solutions (not set by default). The letter L or R stands for Love or Rayleigh.
%   -group                    Switches to group slowness (default=phase)
%   -s <sampling>             Defines the sampling type:
%                               period     regular sampling in period
%                               frequency  regular sampling in frequency
%                               log        regular sampling in log(frequency) (default)
%   -min <min>                Minimum of range for dispersion curve (default=0.2)
%   -max <max>                Maximum of range for dispersion curve (default=20)
%   -vn <count>               Number of velocity samples (only for -grid, default=100)
%   -vmin <min>               Minimum of range for velocity (only for -grid, default=100)
%   -vmax <max>               Maximum of range for velocity (only for -grid, default=3000)
%   -delta-k                  Computes wavenumber gaps between modes. Number of modes must be at least 2.
%   -half-space <index>       Take Vp and Vs from layer index and compute the Rayleigh phase slowness for a half-space with these Vp and Vs.
% 
% Generic options: [level=4]
%   -h, -help [ARG]           Shows help. ARG may be a level (0,1,2...) or a section keyword. Accepted keywords are: all, html, latex, generic, debug, examples, gpdc
%   -args [MAX]               List the argument history diplaying MAX entries (default=50). Üse '-args 0' to print all entries recorded so far.
%   -rargs [INDEX]            Reuse argument list with INDEX. See '-args' to get the available argument lists.Wihtout INDEX the last argument list is used.
%   -version                  Show version information
%   -app-version              Show short version information
%   -j, -jobs <N>             Allow a maximum of N simulteneous jobs for parallel computations (default=36).
%   -verbosity <V>            Set level of verbosity (default=0)
%   -locale <L>               Set current locale
%   -batch                    Prevent the storage of user history (e.g. -args or recent file menus). Use it when running inside a script, to prevent pollution of user history.
%   -qt-plugin-paths          Print the list of paths where Qt plugins are search.
% 
% Debug options: [level=5]
%   -nobugreport              Does not generate bug reports in case of error.
%   -reportbug                Starts bug report dialog, information about bug is passed through stdin. This option is used internally to report bugs if option -nobugreport is not specified.
%   -warning-critical         Consider warnings as critical, stops execution and issue a bug report. Mainly used for debug, to be able to start the debugger even after a warning.
%   -reportint                Starts bug report dialog, information about interruption is passed through stdin. This option is used internally to report interruptions if option -nobugreport
%                             is not specified.
%   -sleep <S>                Sleep for S seconds at the beginning to let, for instance, a debugger to connect.
% 
% Examples:
% 
%          gpdc < test.model
% 
%   Calculate fundamental Rayleigh dispersion curve from 0.2 Hz to 20 Hz for model 'test.model'.
% 
%          gpdc -L 1 -R 2 < test.model
% 
%   Calculate fundamental Love mode and fundamental and first higher mode for Rayleigh.
% 
%          gpdc < test.model | figue -c
% 
%   Calculate the same dispersion curve and plot it.
% 
%          gpdc < test.model | figue -c -m dc.mkup
% 
%   Show the same dispersion curve on a log-log plot. 'dc.mkup' is a tar.gz file containing an xml description of the graphic format, it can be generated from figue's interface.
% 
% See also:
%   gpell, gpprofile
%   More information at http://www.geopsy.org
% 
% Authors:
%   Marc Wathelet
%   Marc Wathelet (LGIT, Grenoble, France)


end
