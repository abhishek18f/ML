function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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


z =X*theta;
h = zeros(size(z));

for q = 1:(size(z))[1];
     h(q) = 1/(1+exp(-z(q)));
endfor

theta2 = theta.*theta;
sum_0 = sum(theta);
sum_02  = sum(theta2) - theta2(1);
J = -1*(y'*log(h) + (1-y)'*log(1-h))/m + lambda*sum_02/(2*m);

grad  = (X'*(h-y))/m + lambda*theta/m;
grad(1) = ((X'*(h-y))/m)(1);




% =============================================================

end
