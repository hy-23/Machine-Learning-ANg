function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
minError = Inf;
error_local = Inf;
steps = [0.01 0.03 0.1 0.3 1 3 10 30];

for i = 1:8,
	C_local = steps(i);
	for j = 1:8,
		sigma_local = steps(j);
		
		%train the data using X and y.
		model = svmTrain(X, y, C_local, @(x1, x2) gaussianKernel(x1, x2, sigma_local));
		
		%use the trained model to make predictions on Xval.
		predictions = svmPredict(model, Xval);
		
		% comparing the predicted value with the known value yval.
		error_local = mean(double(predictions ~= yval));
		
		% err((8*(i-1))+j) = error_local; % if required for debugging you can store all the 64 error values.
		if(error_local < minError),
			minError = error_local;
			C = C_local;
			sigma = sigma_local;
		end;
	end;
end;
% at the end of 64 iterations, you will have C and sigma for the minimumest error.
		






% =========================================================================

end
