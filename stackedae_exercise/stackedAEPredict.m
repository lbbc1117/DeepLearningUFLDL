function [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:(hiddenSize+1)*numClasses), numClasses, hiddenSize+1);

% Extract out the "stack"
stack = params2stack(theta((hiddenSize+1)*numClasses+1:end), netconfig);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

m = size(data,2);
Z2 = stack{1}.w*data + repmat(stack{1}.b,1,m);
A2 = sigmoid(Z2);
Z3 = stack{2}.w*A2 + repmat(stack{2}.b,1,m);
A3 = sigmoid(Z3);
prob = softmaxProbability( softmaxTheta, [A3;ones(1,m)]);
[maxProb, pred] = max(prob);

% -----------------------------------------------------------

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
