function prob = computProbability( theta, data )
% theta k*(n+1)
% data (n+1)*m

prob = theta*data;
prob = bsxfun(@minus, prob, max(prob,[],1));
prob = exp(prob);
prob = bsxfun(@rdivide, prob, sum(prob));

end

