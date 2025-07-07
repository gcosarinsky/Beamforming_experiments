
param.fc = 5e6;
param.fs = 62.5e6;
param.pitch = 0.5e-3;
param.Nelements = 128;
param.c = 6300;
%param.width = 128*0.5e-6;
%param.bandwith = 80;

% load acquisiton and FIR coefficients
load("pwi_acq_25angles.mat")
load('bp_coef.mat')
a = double(a);

% pixel grid
[xi,zi] = meshgrid(linspace(-20e-3,20e-3,200),linspace(0, 30e-3,200));

% some parameters
n_samples = size(a, 1);
n_angles = size(angles, 2);
sig_size = size(a, [1, 2]);

%%
m = {}; % cell for DAS matrixes
disp('Compute DAS matrix for each angle and concatenate')
tic
for i=1:n_angles
    dly = txdelay(param, angles(i));
    m{i} = dasmtx(sig_size, xi, zi, dly, param);
end

m = horzcat(m{:});
toc

%% pasabanda y hilbert
disp('compute bandpass filter and hilbert')
tic
s = filtfilt(bp_coef, 1, a);
toc

tic
sh = hilbert(s); 
toc

%%
disp('beamform')
tic
bf = m*sh(:);
toc

%%

figure
imagesc(reshape(abs(bf), size(xi)))

%% compute excecution times
t.fir = timeit(@() filtfilt(bp_coef, 1, a), 1);
t.hilbert = timeit(@() hilbert(s), 1);
t.bf = timeit(@() m*sh(:), 1);