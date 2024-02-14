% This is first implementation of GRAPPA on 5 Channel data. Results are implemented for all 5 channels combined. Try implementation for 16 or 32 channel data.

clear;clc;

brainCoilsData = load('brain_coil.mat');
brainCoils = brainCoilsData.brain_coil_tmp;
[phaseLength, freqLength, numCoils] = size(brainCoils);


%% Show original Brain Image
%%
originalImage = rsos(brainCoils);
% originalImage = brainCoils(:,:,3);


imagesc(originalImage);
title('Original Image');
colormap('gray');

fullKspace = (fftshift(fft2(brainCoils)));

figure;
imagesc(rsos(fullKspace), [0, 100000]);
title('Original k-space');
colormap('gray');

%% Setting parameters for later use.
% Due to overlap with the downsampling lines, the actual ACS length may be 1 
% or 2 lines longer than specified.

samplingRate = 2;
acsLength = 12;
kernelHeight = 4;
kernelWidth = 3;


%% Making the mask
% Creating a mask for the downsampling and the acs lines.

mask = zeros(size(fullKspace));  % Making a mask for the brain coils.

acsStart = floor((phaseLength - acsLength) ./ 2);
acsFinish = acsStart + acsLength;

for idx=1:phaseLength
    if (acsStart < idx) && (idx <= acsFinish)
        mask(idx, :, :) = 1;
    end

    if mod(idx, samplingRate) == 0
        mask(idx, :, :) = 1;
    end
end


%% Displaying mask.

figure;
imagesc(mean(mask, 3));
title('Mask array');
colormap('gray');
% colorbar();

%% Generating down-sampled k-space
% Obtain the Hadamard product (elementwise matrix multiplication) between the 
% full k-space data and the mask.

downSampledKspace = fullKspace .* mask;

imagesc(rsos(downSampledKspace), [0, 100000]);
title('Downsampled k-space');
colormap('gray');

%% Check Downsampled Image
%%
dsImage = ifft2(ifftshift(downSampledKspace));
dsImage = rsos(dsImage);

imagesc(dsImage);
title('Downsampled Image');
colormap('gray');


%% Find True ACS
% Find the length of the true ACS domain, including lines due to downsampling.

finder = any(downSampledKspace, [2, 3]);

acsFinder = zeros(phaseLength, 1);
for idx=2:phaseLength-1
    if finder(idx-1) == 1 && finder(idx) == 1 && finder(idx+1) == 1
        acsFinder(idx-1:idx+1) = 1;
    end
end

% Getting the idices of the ACS lines.
acsLines = find(acsFinder);

% Checking whether the parameters fit.
matrixHeight = samplingRate .* (kernelHeight-1) + 1;
acsTrueLength = length(acsLines);

%% Building the weight matrix
% Has equation of X * w = Y in mind.
% 
% X = inMatrix, Y = outMatrix, w = weights.
% 
% Each kernel adds one row to the X and Y matrices.

acsPhases = acsTrueLength - matrixHeight + 1;
numKernels = acsPhases .* freqLength;
kernelSize = kernelHeight .* kernelWidth .* numCoils;
outSize = numCoils .* (samplingRate - 1);

inMatrix = zeros(numKernels, kernelSize);
outMatrix = zeros(numKernels, outSize);
hkw = floor(kernelWidth/2);  % Half kernel width.
hkh = kernelHeight/2;  % Half kernel height.
kidx = 1;  % "Kernel index" for counting the number of kernels.
for acsLine=acsLines(1:acsPhases)'
    phases = linspace(acsLine, acsLine+matrixHeight-1, kernelHeight);  % Phases of the kernel
    for freq=1:freqLength
        freqs = linspace(freq-hkw, freq+hkw, kernelWidth);  % Frequencies of the kernel.
        freqs = mod(freqs-1, freqLength) + 1;  % For circular indexing.
        
        selected = linspace(phases(hkh)+1, phases(hkh+1)-1, samplingRate-1);
        selected = mod(selected-1, phaseLength) + 1;  % Selected Y phases.
        
        tempX = downSampledKspace(phases, freqs, :);
        tempY = downSampledKspace(selected, freq, :);
        
        % Filling in the matrices row by row.
        inMatrix(kidx, :) = reshape(tempX, 1, kernelSize);
        outMatrix(kidx, :) = reshape(tempY, 1, outSize);
        
        kidx = kidx + 1;
    end
end

weights = pinv(inMatrix) * outMatrix;  % Calculate the weight matrix.


%% GRAPPA Reconstruction
% Performing a naive reconstruction according to first principles causes an 
% overlap problem.
% 
% The lines immediately before and after the ACS lines are not necessarily 
% spaced with the sampling rate as the spacing.
% 
% This causes alteration of the original data if GRAPPA reconstruction is 
% performed naively.
% 
% The solution is to perform reconstruction on a blank, and then overwrite 
% all vlaues with the original data.
% 
% This alleviates the problem of having to do special operations for the 
% values at the edges.
% 
% Also, the lines of k-space at the start or end of k-space may be neglected 
% (depending on the implementation).
% 
% This requires shifting the finder by the sampling rate to look at the phase 
% from one step above.
% 
% If the downsampling does not match due to incorrect k-space dimensions, 
% errors will be overwritten by the final correction process.

% Find the indices to fill, including at the beginning of k-space.
temp1 = find(diff(finder) == -1) + 1;  % Gets nearly all the lines.
temp2 = find(diff(circshift(finder, samplingRate)) == -1) - samplingRate + 1;
% Second line exists to catch the first few empty lines,
% if any are present.
fillFinder = unique([temp1; temp2]);

% Shift from first fill line to beginning of kernel data.
upShift = (hkh-1) .* samplingRate + 1;
% Shift from first fill line to end of kernel data.
downShift = hkh .* samplingRate - 1;

grappaKspace = zeros(size(downSampledKspace));

for phase=fillFinder'
    phases = linspace(phase-upShift, phase+downShift, kernelHeight);
    phases = mod(phases-1, phaseLength) + 1;  % Circularly indexed phases.
    for freq=1:freqLength
        freqs = linspace(freq-hkw, freq+hkw, kernelWidth);
        freqs = mod(freqs-1, freqLength) + 1;  % Circularly indexed frequencies.
        
        kernel = downSampledKspace(phases, freqs, :);
        % One line of the input matrix.
        tempX = reshape(kernel, 1, kernelSize);
        
        % One line of the output matrix.
        tempY = tempX * weights;
        tempY = reshape(tempY, (samplingRate-1), 1, numCoils);
        
        % Selected lines of the output matrix to be filled in.
        selected = linspace(phases(hkh)+1, phases(hkh+1)-1, samplingRate-1);
        selected = mod(selected-1, phaseLength) + 1;
        
        grappaKspace(selected, freq, :) = tempY;
    end
end

% Filling in all the original data.
% Doing it this way solves the edge overlap problem.
grappaKspace(finder, :, :) = downSampledKspace(finder, :, :);


%% Display recon image
%%
recon = ifft2((ifftshift(grappaKspace)));

reconImage = rsos(recon);

imagesc(reconImage);
title('Reconsrtucted Image');
colormap('gray');


%% Display difference image
%%
% deltaImage = reconImage - originalImage;
% 
% imagesc(angle(deltaImage));
% title('Difference Image');
% colormap('gray');

%% Performing grappa on reconstructed part 

%% Making the alternate mask

mask_2 = zeros(size(fullKspace));  

acsStart = floor((phaseLength - acsLength) ./ 2);
acsFinish = acsStart + acsLength;

for idx=1:phaseLength
    if (acsStart < idx) && (idx <= acsFinish)
        mask_2(idx, :, :) = 1;
    end

    if mod(idx, samplingRate) == 1
        mask_2(idx, :, :) = 1;
    end
end

figure;
imagesc(mean(mask_2, 3));
title('Mask 2 array');
colormap('gray');


%% Generating down-sampled grappa k-space
downSampledKspace_2 = grappaKspace .* mask_2;
imagesc(rsos(downSampledKspace), [0, 100000]);
title('Downsampled k-space');
colormap('gray');


%% Find True ACS
% Find the length of the true ACS domain, including lines due to downsampling.

finder = any(downSampledKspace_2, [2, 3]);

acsFinder = zeros(phaseLength, 1);
for idx=2:phaseLength-1
    if finder(idx-1) == 1 && finder(idx) == 1 && finder(idx+1) == 1
        acsFinder(idx-1:idx+1) = 1;
    end
end

% Getting the idices of the ACS lines.
acsLines = find(acsFinder);

% Checking whether the parameters fit.
matrixHeight = samplingRate .* (kernelHeight-1) + 1;
acsTrueLength = length(acsLines);

%% Building the weight matrix
% Has equation of X * w = Y in mind.
% 
% X = inMatrix, Y = outMatrix, w = weights.
% 
% Each kernel adds one row to the X and Y matrices.

acsPhases = acsTrueLength - matrixHeight + 1;
numKernels = acsPhases .* freqLength;
kernelSize = kernelHeight .* kernelWidth .* numCoils;
outSize = numCoils .* (samplingRate - 1);

inMatrix = zeros(numKernels, kernelSize);
outMatrix = zeros(numKernels, outSize);
hkw = floor(kernelWidth/2);  % Half kernel width.
hkh = kernelHeight/2;  % Half kernel height.
kidx = 1;  % "Kernel index" for counting the number of kernels.
for acsLine=acsLines(1:acsPhases)'
    phases = linspace(acsLine, acsLine+matrixHeight-1, kernelHeight);  % Phases of the kernel
    for freq=1:freqLength
        freqs = linspace(freq-hkw, freq+hkw, kernelWidth);  % Frequencies of the kernel.
        freqs = mod(freqs-1, freqLength) + 1;  % For circular indexing.
        
        selected = linspace(phases(hkh)+1, phases(hkh+1)-1, samplingRate-1);
        selected = mod(selected-1, phaseLength) + 1;  % Selected Y phases.
        
        tempX = downSampledKspace_2(phases, freqs, :);
        tempY = downSampledKspace_2(selected, freq, :);
        
        % Filling in the matrices row by row.
        inMatrix(kidx, :) = reshape(tempX, 1, kernelSize);
        outMatrix(kidx, :) = reshape(tempY, 1, outSize);
        
        kidx = kidx + 1;
    end
end

weights_2 = pinv(inMatrix) * outMatrix;  % Calculate the weight matrix.


%% GRAPPA Reconstruction

temp1 = find(diff(finder) == -1) + 1;  % Gets nearly all the lines.
temp2 = find(diff(circshift(finder, samplingRate)) == -1) - samplingRate + 1;

fillFinder = unique([temp1; temp2]);

% Shift from first fill line to beginning of kernel data.
upShift = (hkh-1) .* samplingRate + 1;
% Shift from first fill line to end of kernel data.
downShift = hkh .* samplingRate - 1;

grappaKspace_2 = zeros(size(downSampledKspace_2));

for phase=fillFinder'
    phases = linspace(phase-upShift, phase+downShift, kernelHeight);
    phases = mod(phases-1, phaseLength) + 1;  % Circularly indexed phases.
    for freq=1:freqLength
        freqs = linspace(freq-hkw, freq+hkw, kernelWidth);
        freqs = mod(freqs-1, freqLength) + 1;  % Circularly indexed frequencies.
        
        kernel = downSampledKspace_2(phases, freqs, :);
        % One line of the input matrix.
        tempX = reshape(kernel, 1, kernelSize);
        
        % One line of the output matrix.
        tempY = tempX * weights_2;
        tempY = reshape(tempY, (samplingRate-1), 1, numCoils);
        
        % Selected lines of the output matrix to be filled in.
        selected = linspace(phases(hkh)+1, phases(hkh+1)-1, samplingRate-1);
        selected = mod(selected-1, phaseLength) + 1;
        
        grappaKspace_2(selected, freq, :) = tempY;
    end
end

% Filling in all the original data.
% Doing it this way solves the edge overlap problem.
grappaKspace_2(finder, :, :) = downSampledKspace_2(finder, :, :);



%%

% imagesc(squeeze(abs(grappaKspace(:,:,1))))

imagesc(abs(ifft2((ifftshift((squeeze(grappaKspace(:,:,1))))))))

%% Each channel reconstruction
% figure;
% subplot(2,5,1)
% imagesc(abs(ifft2((ifftshift((squeeze(grappaKspace(:,:,1))))))))
% title("Reconstructed images")
% 
% subplot(2,5,2)
% imagesc(abs(ifft2((ifftshift(abs(fullKspace(:,:,1)))))))
% title("Original images")
% 
% subplot(2,5,3)
% imagesc(abs(ifft2((ifftshift((squeeze(grappaKspace(:,:,2))))))))
% title("Reconstructed images")
% 
% subplot(2,5,4)
% imagesc(abs(ifft2((ifftshift(abs(fullKspace(:,:,2)))))))
% title("Original images")
% 
% subplot(2,5,5)
% imagesc(abs(ifft2((ifftshift((squeeze(grappaKspace(:,:,3))))))))
% title("Reconstructed images")
% 
% subplot(2,5,6)
% imagesc(abs(ifft2((ifftshift(abs(fullKspace(:,:,3)))))))
% title("Original images")
% 
% 
% subplot(2,5,7)
% imagesc(abs(ifft2((ifftshift((squeeze(grappaKspace(:,:,4))))))))
% title("Reconstructed images")
% 
% subplot(2,5,8)
% imagesc(abs(ifft2((ifftshift(abs(fullKspace(:,:,4)))))))
% title("Original images")
% 
% 
% subplot(2,5,9)
% imagesc(abs(ifft2((ifftshift((squeeze(grappaKspace(:,:,5))))))))
% title("Reconstructed images")
% 
% subplot(2,5,10)
% imagesc(abs(ifft2((ifftshift(abs(fullKspace(:,:,5)))))))
% title("Original images")
% 


%% Display recon image
%%
recon_2 = ifft2((ifftshift(grappaKspace_2)));
temp = recon_2(:, :, 1);
reconImage_2 = rsos(recon_2);

imagesc(reconImage_2);
title('Reconstructed applied on Grappa output ');
colormap('gray');

%% Summary 1
%%
subplot(2, 2, 1);
imagesc(originalImage);
title('Original Image');
colormap('gray');
axis('off');

subplot(2, 2, 2);
imagesc(dsImage);
title('Down-Sampled Image');
colormap('gray');
axis('off');

subplot(2, 2, 3);
imagesc(reconImage);
title('Reconstructed Image');
colormap('gray');
axis('off');

subplot(2, 2, 4);
imagesc(reconImage_2);
title('Reconstructed applied on Grappa output');
colormap('gray');
axis('off');



ssimval_1 = ssim(reconImage,originalImage);
fprintf('\n The SSIM value of Grappa reconstruction is %0.4f', ssimval_1);

ssimval_2 = ssim(reconImage_2,originalImage);
fprintf('\n The SSIM value of Reconstruction applied on grappa output %0.4f \n', ssimval_2);


%% RSOS 
%%
function image = rsos(data)
% RSOS Root Sum of Squares Function.
% 
% The root Sum of Squares function necessary for converting 3D multichannel 
% complex data into 2D real valued data.
image = abs(data) .^ 2;
image = sum(image, 3);
image = sqrt(image);
end
