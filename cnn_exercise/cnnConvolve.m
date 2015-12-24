function convolvedFeatures = cnnConvolve(patchDim, numFeatures, images, W, b, ZCAWhite, meanPatch)
%cnnConvolve Returns the convolution of the features given by W and b with
%the given images
%
% Parameters:
%  patchDim - patch (feature) dimension
%  numFeatures - number of features
%  images - large images to convolve with, matrix in the form
%           images(r, c, channel, image number)
%  W, b - W, b for features from the sparse autoencoder
%  ZCAWhite, meanPatch - ZCAWhitening and meanPatch matrices used for
%                        preprocessing
%
% Returns:
%  convolvedFeatures - matrix of convolved features in the form
%                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)

numImages = size(images, 4);
imageDim = size(images, 1);
imageChannels = size(images, 3);

convolvedFeatures = zeros(numFeatures, numImages, imageDim - patchDim + 1, imageDim - patchDim + 1);

% Instructions:
%   Convolve every feature with every large image here to produce the 
%   numFeatures x numImages x (imageDim - patchDim + 1) x (imageDim - patchDim + 1) 
%   matrix convolvedFeatures, such that 
%   convolvedFeatures(featureNum, imageNum, imageRow, imageCol) is the
%   value of the convolved featureNum feature for the imageNum image over
%   the region (imageRow, imageCol) to (imageRow + patchDim - 1, imageCol + patchDim - 1)
%
% Expected running times: 
%   Convolving with 100 images should take less than 3 minutes 
%   Convolving with 5000 images should take around an hour
%   (So to save time when testing, you should convolve with less images, as
%   described earlier)

% -------------------- YOUR CODE HERE --------------------
% Precompute the matrices that will be used during the convolution. Recall
% that you need to take into account the whitening and mean subtraction
% steps

% 由于之前使用了白化数据训练SAE，现在做卷积特征提取依然要使用白化数据
W_ZCAWhite = W*ZCAWhite;
b_ZCAWhite = b - W_ZCAWhite*meanPatch;

patchSize = patchDim^2;
% 将2维的参数矩阵改造成3维，将颜色通道作为一个维度提取出来，方便后续计算
W_ZCAWhite = reshape(W_ZCAWhite,numFeatures,patchSize,imageChannels);

% --------------------------------------------------------

convolvedFeatures = zeros(numFeatures, numImages, imageDim - patchDim + 1, imageDim - patchDim + 1);
for imageNum = 1:numImages
  for featureNum = 1:numFeatures

    % convolution of image with feature matrix for each channel
    convolvedImage = zeros(imageDim - patchDim + 1, imageDim - patchDim + 1);
    for channel = 1:3

      % Obtain the feature (patchDim x patchDim) needed during the convolution
      % ---- YOUR CODE HERE ----
      feature = W_ZCAWhite(featureNum,:,channel);   % 每次只对一个通道做卷积  
      feature = reshape(feature,patchDim,patchDim); % 卷积窗口是矩阵（不是向量）
      % ------------------------

      % Flip the feature matrix because of the definition of convolution,
      % as explained later
      feature = rot90(squeeze(feature),2);
      
      % Obtain the image
      im = squeeze(images(:, :, channel, imageNum));

      % Convolve "feature" with "im", adding the result to convolvedImage
      % be sure to do a 'valid' convolution
      % ---- YOUR CODE HERE ----
      % 将3个颜色通道的卷积相加得到最终的卷积特征
      convolvedImage = convolvedImage + conv2(im,feature,'valid');         
      % ------------------------

    end
    
    % Subtract the bias unit (correcting for the mean subtraction as well)
    % Then, apply the sigmoid function to get the hidden activation
    % ---- YOUR CODE HERE ----
    convolvedImage = convolvedImage + b_ZCAWhite(featureNum);    
    convolvedImage = sigmoid(convolvedImage);
    % ------------------------
    
    % The convolved feature is the sum of the convolved values for all channels
    convolvedFeatures(featureNum, imageNum, :, :) = convolvedImage;
  end
end

end


function sigm = sigmoid(x)
    sigm = 1./(1+exp(-x));
end

