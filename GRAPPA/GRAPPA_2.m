% another implemetation of GRAPPA on 32 channel dataset. Here K space is of size 96 x 96 and ACS region is given of size 32 x 92. 

clear;clc;

input   =   matfile('data.mat');
truth   =   input.truth;
calib   =   input.calib;

% calib_reshape = permute(calib,[2,3,1]);
% truth_reshape = permute(truth,[2,3,1]);
% 
% kspace_calib = squeeze(calib_reshape(:,:,4));
% kspace_truth = squeeze(truth_reshape(:,:,4));
% 
% 
% calib_image = abs(ifftshift(ifft2(kspace_calib)));
% truth_image = abs(ifftshift(ifft2(kspace_truth)));

% imagesc(truth_image)
% imagesc(abs(calib_reshape(:,:,3)))
% imagesc(abs(truth_reshape(:,:,3)))

%%
R       =   [1,2];
kernel  =   [3,4];

mask    =   false(32,96,96);
mask(:,:,1:2:end)   =   true;

data    =   truth.*mask;


recon   =   grappa(data, calib, R, kernel);
show_quad(data, recon, 'R=2');

%% All functions functions 

for i=1:32
    
    imagesc


end


%% Grappa main function

function recon = grappa(data, calib, R, kernel)
    %   Check if R is scalar and set Rx=1
    if isscalar(R)
        R   =   [1, R];
    end
    
    %  Pad data to deal with kernels applied at k-space boundaries
    pad     =   grappa_get_pad_size(kernel, R);
    pdata   =   grappa_pad_data(data, pad);
    
    %   Define and pad the sampling mask, squeeze it because we don't need the coil dimension
    mask    =   grappa_pad_data(data~=0, pad);
    
    %  Loop over R-1 different kernel types
    for type = 1:R(2)-1
        %  Collect source and target calibration points for weight estimation
        [src_calib, trg_calib]  =   grappa_get_indices(kernel, true(size(calib)), pad, R, type);
    
        %  Perform weight estimation
        weights =   grappa_estimate_weights(calib, src_calib, trg_calib);
    
        %  Collect source points in under-sampled data for weight application
        [src, trg]  =   grappa_get_indices(kernel, circshift(mask,type,3), pad, R, type);
    
        %  Apply weights to reconstruct missing data
        pdata   =  grappa_apply_weights(pdata, weights, src, trg);
    end

    recon   =   grappa_unpad_data(pdata, pad);
end

%% Other functions

function pad = grappa_get_pad_size(kernel, R)
    %   Compute size of padding needed in each direction
    pad =   floor(R.*kernel/2);
end



function pdata = grappa_pad_data(data, pad)

    %   Zero-Pad
    %   Apply zero-padding to kx, ky directions (don't pad coil dimension)
    pdata   =   padarray(data, [0 pad]);
    
    %   Cyclic-Pad
    %   Here we additionally copy data so that k-space has cyclic boundary conditions
    %   This isn't absolutely necessary - if you like, just leave it zero-padded
    %   First pad left boundary 
    pdata(:, 1:pad(1),:) =   pdata(:,1+size(pdata,2)-2*pad(1):size(pdata,2)-pad(1),:);
    
    %   Next pad right boundary
    pdata(:, size(pdata,2)-pad(1)+1:size(pdata,2),:) =   pdata(:,pad(1)+1:2*pad(1),:);
    
    %   Third, pad top boundary 
    pdata(:,:,1:pad(2)) =   pdata(:,:,1+size(pdata,3)-2*pad(2):size(pdata,3)-pad(2));
    
    %   Finally, pad bottom boundary
    pdata(:,:,size(pdata,3)-pad(2)+1:size(pdata,3)) =   pdata(:,:,pad(2)+1:2*pad(2));

end

function [src, trg] = grappa_get_indices(kernel, samp, pad, R, type)

    %   Get dimensions
    dims    =   size(samp);
    
    %   Make sure the under-sampling is in y-only
    %   There are a few things here that require that assumption
    if R(1) > 1
        error('x-direction must be fully sampled');
    end
    
    %   Make sure the kernel is odd in x, and even in y
    if mod(kernel(1),2)==0 || mod(kernel(2),2)==1
        error('Kernel geometry is not allowed');
    end
    
    %   Make sure the type parameter makes sense
    %   It should be between 1 and R-1 (inclusive)
    if type < 1 || type > R(2)-1
        error('Type parameter is inconsistent with R');
    end
    
    %   To get absolute kernel distances, multiply kernel and R
    kernel  =   kernel.*R;
    
    %   Find the limits of all possible target points given padding
    kx  =   1+pad(1):dims(2)-pad(1);
    ky  =   1+pad(2):dims(3)-pad(2);
    
    %  Compute indices for a single coil
    
    %   Find relative indices for kernel source points
    mask    =   false(dims(2:3));
    mask(1:R(1):kernel(1), 1:R(2):kernel(2))    =   true;
    k_idx   =   find(mask);
    
    %   Find the index for the desired target point (depends on type parameter)
    %   To simply things, we require than kernel size in x is odd
    %   and that kernel size in y is even
    mask    =   false(dims(2:3));
    mask((kernel(1)+1)/2, (kernel(2)/2-R(2)+1)+type)    =   true;
    k_trg   =   find(mask);
    
    %   Subtract the target index from source indices
    %   to get relative linear indices for all source points
    %   relative to the target point (index 0, target position)
    k_idx   =   k_idx - k_trg;
    
    %   Find all possible target indices
    mask    =   false(dims(2:3));
    mask(kx,ky) =   squeeze(samp(1,kx,ky));
    trg     =   find(mask); 
    
    %   Find all source indices associated with the target points in trg
    src =   bsxfun(@plus, k_idx, trg');
    
    %  Now replicate indexing over all coils
    
    %   Final shape of trg should be (#coils, all possible target points)
    trg =   (trg'-1)*dims(1)+1;
    trg =   bsxfun(@plus, trg, (0:dims(1)-1)');
    
    %   Final shape of src should be (#coils*sx*sy, all possible target points)
    src =   (src-1)*dims(1)+1;
    src =   bsxfun(@plus, src(:)', (0:dims(1)-1)');
    src =   reshape(src,[], size(trg,2));

end

function weights = grappa_estimate_weights(calib, src_idx, trg_idx, lambda)

    %   If no lambda provided, don't regularise
    if nargin < 4
        lambda = 0;
    end
    
    %   Collect source and target points based on provided indices
    src     =   calib(src_idx);
    trg     =   calib(trg_idx);
    
    %   Least squares fit for weights
    %weights =   trg*pinv(src);
    weights =   trg*src'*inv(src*src' + norm(src)*lambda*eye(size(src,1)));
end 


function data = grappa_apply_weights(data, weights, src_idx, trg_idx)

    %   Collect source and target points based on provided indices
    src     =   data(src_idx);
    
    %   Apply weights and insert synthesized target points into data
    data(trg_idx)  =   weights*src;

end

function data = grappa_unpad_data(pdata, pad)
    
    %   Subselect inner data absent the padded points
    data    =   pdata(:,pad(1)+1:size(pdata,2)-pad(1), pad(2)+1:size(pdata,3)-pad(2));

end

