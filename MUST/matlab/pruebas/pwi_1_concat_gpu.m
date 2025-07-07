
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
%% trasnfer to gpu
disp('trasnfer DAS matrix to gpu')
tic
m_gpu = gpuArray(m);
wait(gpuDevice);
toc

%% pasabanda y hilbert
disp('transfer signals to gpu')
tic
a_gpu = gpuArray(a);
wait(gpuDevice);
toc

disp('trasnfer FIR coefficients to gpu')
bp_coef_gpu = gpuArray(bp_coef);

disp('compute bandpass filter and hilbert in gpu')
tic
s_gpu = filtfilt(bp_coef_gpu, 1, a_gpu);
wait(gpuDevice);
toc

tic
sh_gpu = hilbert(s_gpu); 
wait(gpuDevice);
toc

%%
disp('beamform in gpu')
tic
bf_gpu = m_gpu*sh_gpu(:);
wait(gpuDevice);
toc

%%

figure
imagesc(reshape(abs(bf_gpu), size(xi)))

%% compute excecution times
t_gpu.signal2gpu = gputimeit(@() gpuArray(a) , 1);
t_gpu.fir = gputimeit(@() filtfilt(bp_coef, 1, a), 1);
t_gpu.hilbert = gputimeit(@() hilbert(s_gpu), 1);
t_gpu.bf = gputimeit(@() m*sh_gpu(:), 1);

%%
function bf = do_bf(a, ) 

%%
n = 20;
a_rand = rand([size(a) n]) ;


tic
for i=1:n
    
end