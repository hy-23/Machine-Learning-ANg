function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

predict = X * theta;
cost = (predict - y) .* (predict - y);
reg = theta .* theta;

p1 = sum(cost(:));
p2 = sum(reg(:)) - (theta(1,1)*theta(1,1));

J = (1/(2*m)) * (p1 + (lambda * p2));

error = predict - y;
grad = (1/m)*(X' * error);

temp = grad(1,1);
grad = grad + ((lambda/m) * theta);
grad(1,1) = temp;








% =========================================================================

grad = grad(:);

end
