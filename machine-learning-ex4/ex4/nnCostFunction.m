function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1:
	aLAyer1 = [ones(m,1) X];
	z2 = (aLAyer1 * Theta1');
	aLAyer2 = sigmoid(z2);
	aLAyer2 = [ones(m,1) aLAyer2];
	z3 = (aLAyer2 * Theta2');
	aLAyer3 = sigmoid(z3); % aLAyer3 is now a matrix of 5000 x 10

	% However, y is a matrix of 5000 x 1. This needs to be converted to a matrix of
	% 5000 x 10. Which is done as below.

	result = full(sparse(1:numel(y), y+1, 1, numel(y), (num_labels+1)));
	% This gives indexing from 0, but for us it has to start from 1.
	% Hence, obtained a matrix of 5000 x 11, where 1st column represents
	% 0. Thus, to adapt for our system of representation, where 1 is 1st 
	% column and 10 is 10th column - Delete the first column.

	result(:,[1]) = [];

	% Now, compute cost by doing element wise multiplication.
	param1 = result     .* log(aLAyer3);
	param2 = (1-result) .* log(1-aLAyer3);
	cost = param1 + param2;

	% Cost is also a 5000 x 10 matrix. 5000 training samples and 10 number of
	% neurons at the end.

	% For actual cost, we need sum of errors across all units and all training set.
	% Thus sum of all elements in the cost matrix would give us the number.

	J = (-1/m)*(sum(cost(:)));

% Part 2:

	z2 = [ones(m,1) z2];

	delta3 = -result + aLAyer3; 

	dz2 = sigmoidGradient(z2);
	delta2 = (delta3 * Theta2) .* (dz2); 
	delta2(:,[1]) = [];
	Theta1_grad = (1/m)*(delta2' * aLAyer1);
	Theta2_grad = (1/m)*(delta3' * aLAyer2);

% Part 3:
	Theta1_reg = Theta1;
	Theta2_reg = Theta2;
	Theta1_reg(:,[1]) = []; % DO NOT REGULARIZE the weight for the bias.
	Theta2_reg(:,[1]) = []; % DO NOT REGULARIZE the weight for the bias.
	
	sqTheta1 = Theta1_reg .* Theta1_reg;
	sqTheta2 = Theta2_reg .* Theta2_reg;
	ssqTheta1 = sum(sqTheta1(:));
	ssqTheta2 = sum(sqTheta2(:));
	Jreg = (lambda / (2*m)) * (ssqTheta1 + ssqTheta2);
	J += Jreg;
	
	
	% for gradient calculations to be vectorized, prepare a matrix
	% which when added to Theta1_grad adds 0 to the first column.
	% These first columns are the weights corresponding to the bias.
	n = size(Theta1_grad, 1);
	Theta1_reg = [zeros(n,1) Theta1_reg];
	
	n = size(Theta2_grad, 1);
	Theta2_reg = [zeros(n,1) Theta2_reg];

	Theta1_grad += ((lambda/m) * Theta1_reg);
	Theta2_grad += ((lambda/m) * Theta2_reg);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
