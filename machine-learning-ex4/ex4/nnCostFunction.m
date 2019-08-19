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

ymat = zeros(m, num_labels);
for b = 1:m,
  ymat(b, y(b)) = 1;
endfor
for i = 1:m,
  xval = X(i,:)';
  secondlayer = sigmoid(Theta1*[1;xval]);
  output = sigmoid(Theta2*[1;secondlayer]);
    J = J+sum(-(ymat(i,:))'.*log(output)-(1-(ymat(i,:))').*log(1-output));
 
  
endfor
J = J/m;
newtheta1 = Theta1;
newtheta1(:,1) = zeros(size(Theta1,1),1);
newtheta2 = Theta2;
newtheta2(:,1) = zeros(size(Theta2,1),1);
J = J + (lambda/(2*m))*(sum(sum(newtheta1.^2))+sum(sum(newtheta2.^2)));
for t = 1:m,
  
  xval = X(t,:)';
    secondlayer = sigmoid(Theta1*[1;xval]);
    output = sigmoid(Theta2*[1;secondlayer]);
    deltaone = output - ymat(t,:)';
  
    deltatwo = (Theta2'*deltaone);
    deltatwo = deltatwo(2:end);
    deltatwo = deltatwo.*sigmoidGradient(Theta1*[1;xval]);

    Theta1_grad = Theta1_grad+deltatwo*[1;xval]';
    Theta2_grad = Theta2_grad + deltaone*[1;secondlayer]';
endfor

Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;
newgrad1 = zeros(size(Theta1_grad));
newgrad2 = zeros(size(Theta2_grad));
newgrad1 = newgrad1 + (lambda/m)*Theta1;
newgrad1(:,1) = zeros(size(newgrad1,1),1);
newgrad2 = newgrad2 + (lambda/m)*Theta2;
newgrad2(:,1) = zeros(size(newgrad2,1),1);
Theta1_grad = Theta1_grad + newgrad1;
Theta2_grad = Theta2_grad + newgrad2;










% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
