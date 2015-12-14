function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);  %columnwise

m = size(data, 2);

groundTruth = full(sparse(labels, 1:m, 1));  % 即数据标记

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

prob = softmaxProbability( theta, data );

cost = (-1/m)*sum(sum(log(prob).*groundTruth)) + (lambda/2)*sum(sum(theta.^2));
thetagrad = (-1/m)*data*(groundTruth-prob)'+lambda*theta';
thetagrad = thetagrad';

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)]; %columnwise
end

