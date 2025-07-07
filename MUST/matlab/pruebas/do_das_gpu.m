function bf = do_das_gpu(a, dasmtx_gpu, bp_coef_gpu)
  a_gpu = gpuArray(a);
  s = filtfilt(bp_coef_gpu, 1, a_gpu);
  sh = hilbert(s); 
  bf = dasmtx_gpu*sh(:);
end
