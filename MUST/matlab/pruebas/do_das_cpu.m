function bf = do_das_cpu(a, dasmtx, bp_coef)
  s = filtfilt(bp_coef, 1, a);
  sh = hilbert(s); 
  bf = dasmtx*sh(:);
end
