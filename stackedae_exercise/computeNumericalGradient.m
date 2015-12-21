function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 

% Number of parameters
n = size(theta);
  
% Initialize numgrad with zeros
numgrad = zeros(n);

%% ---------- YOUR CODE HERE --------------------------------------
% Instructions: 
% Implement numerical gradient checking, and return the result in numgrad.  
% (See Section 2.3 of the lecture notes.)
% You should write code so that numgrad(i) is (the numerical approximation to) the 
% partial derivative of J with respect to the i-th input argument, evaluated at theta.  
% I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
% respect to theta(i).
%                
% Hint: You will probably want to compute the elements of numgrad one at a time. 

EPSILON = 1e-4;

for i = 1:n
    ei = zeros(n);
    ei(i) = 1;
    numgrad(i) = (J(theta+EPSILON*ei) - J(theta-EPSILON*ei)) / (2*EPSILON);
end

%% ---------------------------------------------------------------
end
