%%================================================================
%  Here we provide the code to load natural image data into x.
%  x will be a 144 * 10000 matrix, where the kth column x(:, k) corresponds to
%  the raw image data from the kth 12x12 image patch sampled.
%  You do not need to change the code below.

x = sampleIMAGESRAW();
figure('name','Raw images');
randsel = randi(size(x,2),36,1); % A random selection of samples for visualization
display_network(x(:,randsel));
print -djpeg 1_raw_images.jpg 


% Remove DC (mean of images). 
x = bsxfun(@minus, x, mean(x));
figure('name','Zero mean images');
display_network(x(:,randsel));
print -djpeg 2_zero_mean_images.jpg 

%%================================================================
%% Implement PCA with whitening and regularisation
m = size(x,2);
Sigma = x*x'/m;
[U,S,V] = svd(Sigma);
epsilon = 0.1;
varDividers = diag(diag((S+epsilon).^(-1/2)));
xPCAWhite = varDividers*U'*x;

%%================================================================
%% Implement ZCA whitening
xZCAWhite = U*xPCAWhite;

% -------------------- YOUR CODE HERE -------------------- 

% Visualise the data, and compare it to the raw data.
% You should observe that the whitened images have enhanced edges.
figure('name','PCA whitening filters');
display_network((varDividers*U')');
print -djpeg 3_PCA_whitening_filters.jpg   

figure('name','ZCA whitening filters');
display_network((U*varDividers*U')');
print -djpeg 4_ZCA_whitening_filters.jpg 

figure('name','PCA whitened images');
display_network(xPCAWhite(:,randsel));
print -djpeg 5_PCA_whitened_images.jpg 

figure('name','ZCA whitened images');
display_network(xZCAWhite(:,randsel));
print -djpeg 6_ZCA_whitened_images.jpg  