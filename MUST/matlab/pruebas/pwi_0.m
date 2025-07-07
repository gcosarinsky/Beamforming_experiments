
param.fc = 5e6;
param.fs = 62.5e6;
param.pitch = 0.5e-3;
param.Nelements = 128;
param.c = 6300;
%param.width = 128*0.5e-6;
%param.bandwith = 80;

load("pwi_acq_25angles.mat")
load('bp_coef.mat')

[xi,zi] = meshgrid(linspace(-20e-3,20e-3,200),linspace(0, 30e-3,200));
n_angles = size(angles, 2);
sig_size = size(a, [1, 2]);
m = {};

%%
for i=1:n_angles
    dly = txdelay(param, angles(i));
    m{i} = dasmtx(sig_size, xi, zi, dly, param);
end

%% pasabanda y hilbert
tic
a = filtfilt(bp_coef, 1, double(a));
toc

tic
a = hilbert(a); 
toc

%%
bf = {};

tic
for i=1:n_angles
    s = a(:, :, i);
    bf0 = m{i}*s(:) ;
    bf{i} = reshape(bf0, size(xi));
end
toc

tic
bf_tot = bf{1} ;
for i=2:n_angles
    bf_tot = bf_tot + bf{i} ;
end
toc

%%
figure
imagesc(reshape(abs(bf{1}), size(xi)))

figure
imagesc(reshape(abs(bf_tot), size(xi)))