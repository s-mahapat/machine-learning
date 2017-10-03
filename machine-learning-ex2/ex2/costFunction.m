function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
val = 0;
theta_size = size(theta);
for row = 1:m
    z = theta' * X(row,:)';
    hx = 1 / (1 + e^-z);
    temp1 = y(row) * log(hx);
    temp2 = (1-y(row)) * log(1 - hx);
    val = val + temp1 + temp2;
end
J = -1 * val / m;


for j = 1:theta_size
    val2 = 0;
    for row = 1:m
        z = theta' * X(row,:)';
        hx = 1 / (1 + e^-z);
        val2 = val2 + (hx - y(row)) * X(row, j);
    end
    grad(j) = val2 / m;
end







% =============================================================

end
