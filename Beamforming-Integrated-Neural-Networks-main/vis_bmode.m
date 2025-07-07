function vis_bmode(img,pos_z,pos_x,dyn_range)
    img = hilbert(img); % hilbert after beamforming only good if axial pixel positions are dense
    img = 20*log10(abs(img) + 1e-10); % in case of 0 values
    img_max = max(img(:));

    % simple display for log-envelope b-mode images
    imagesc(pos_x,pos_z,img,[img_max-dyn_range img_max])
    axis tight
    colormap gray
end