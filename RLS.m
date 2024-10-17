function [w, y, e] = RLS(x, d, lambda, M)
    % RLS Recursive Least Squares Algorithm
    % Inputs:
    % x     - Input signal (noisy signal)
    % d     - Desired signal (original signal)
    % lambda - Forgetting factor (0 < lambda < 1)
    % M     - Filter order (number of coefficients)

    if nargin < 3
        lambda = 0.99; % Default forgetting factor
    end
    if nargin < 4
        M = 32; % Default filter order
    end
    
    n = length(x); % Length of input signal
    w = zeros(M, 1); % Initialize filter coefficients
    P = eye(M) * 1000; % Initialize error covariance matrix
    y = zeros(n, 1); % Output signal
    e = zeros(n, 1); % Error signal

    for k = M:n
        x_k = flipud(x(k-M+1:k)); % Current input signal frame (length M)
        y(k) = w' * x_k; % Calculate filter output
        e(k) = d(k) - y(k); % Calculate error
        
        k_k = (P * x_k) / (lambda + x_k' * P * x_k); % Calculate gain vector
        w = w + k_k * e(k); % Update filter coefficients
        
        P = (P - k_k * (x_k' * P)) / lambda; % Update error covariance matrix
    end
end