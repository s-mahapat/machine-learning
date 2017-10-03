function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
[nr, nc] = size(z);
if (nr == 1 && nc == 1)
    g = 1 / (1 + e^-z);
else if(nr > 1)
    for row = 1:nr
        for col = 1:nc
            val = z(row, col);
            g(row, col) = 1 / (1 + e^-val);
        end
    end
endif





% =============================================================

end
