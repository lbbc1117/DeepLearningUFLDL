
function [cost,grad,features] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data)
%
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

m = size(data,2);

% cost function without regularization and spasity penalty
Z2 = W1*data + repmat(b1,1,m);
A2 = sigmoid(Z2);
Z3 = W2*A2 + repmat(b2,1,m);
A3 = Z3;
cost = (1/m)*(1/2)*sum(sum((A3-data).^2));

% Regularization term
regularization = (lambda/2) * (sum(sum(W1.^2))+sum(sum(W2.^2)));

% Sparsity penalty term
averageActivationVec = (1/m)*sum(A2,2);
spasityPenalty = beta * sum(  sparsityParam   * log(   sparsityParam  ./ averageActivationVec )  + ...
                            (1-sparsityParam) * log((1-sparsityParam) ./(1-averageActivationVec)) );
                 
% Add regularization and spasity penalty
cost = cost + regularization + spasityPenalty;

% gradients by Back Propagation Algorithm with regularization and spasity penalty
Delta3 = A3-data;
sparsityTermInDelta2 = repmat(beta*(-sparsityParam./averageActivationVec+(1-sparsityParam)./(1-averageActivationVec)), 1, m);
Delta2 = (W2'*Delta3 + sparsityTermInDelta2).*A2.*(1-A2);

W1grad = Delta2*data'/m + lambda*W1;
W2grad = Delta3*A2'/m + lambda*W2;
b1grad = sum(Delta2,2)/m;
b2grad = sum(Delta3,2)/m;

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x) 
    sigm = 1 ./ (1 + exp(-x));
end

