function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

a_Layer1 = X; % dimenions m x n
a_Layer1 = [ones(m,1) a_Layer1]; % adding bias to the a_Layer1

a_Layer2 = sigmoid(a_Layer1 * Theta1'); % computing a_Layer2
a_Layer2 = [ones(m,1) a_Layer2]; % adding bias to the a_Layer2

a_Layer3 = sigmoid(a_Layer2 * Theta2'); % computing a_Layer3

[max, p] = max(a_Layer3, [], 2);






% =========================================================================


end
