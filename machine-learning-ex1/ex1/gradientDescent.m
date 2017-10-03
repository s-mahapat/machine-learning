function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    theta_0 = theta(1);
    theta_1 = theta(2);
    v1 = 0;
    temp = 0;
    v2 = 0;
    for i = 1:m
        temp = ((theta_0 + (theta_1 * X(i,2))) - y(i)) * 1;
        
        v1 = v1 + temp;
    end
    
    for i = 1:m
        temp = ((theta_0 + (theta_1 * X(i,2))) - y(i)) * X(i,2);
        v2 = v2 + temp;
    end

    delta_0 = v1 / m;
    delta_1 = v2 / m;

    delta = [delta_0; delta_1];

    theta = theta - (alpha * delta);

    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);



end

end
