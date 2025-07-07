
param.fc = 5e6;
param.fs = 62.5e6;
param.pitch = 0.5e-3;
param.Nelements = 128;
param.c = 6300;
%param.width = 128*0.5e-6;
%param.bandwith = 80;

load("pwi_acq_25angles.mat")
s = double(a(:, :, 1));
dly = txdelay(param, angles(1));
[xi,zi] = meshgrid(linspace(-20e-3,20e-3,200),linspace(0, 30e-3,200));
m = dasmtx(size(s), xi, zi, dly, param);

tic
bf = m*s(:) ;
toc

imagesc(reshape(abs(bf), size(xi)))