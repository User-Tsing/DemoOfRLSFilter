function [w, y, e] = RLS(x, d, lambda, M)
    % RLS Recursive Least Squares Algorithm
    % RLS滤波算法
    % Inputs:
    % 输入：
    % x     - Input signal (noisy signal) 输入信号 (含噪信号)
    % d     - Desired signal (original signal) 期望信号 (原始信号)
    % lambda - Forgetting factor (0 < lambda < 1) 遗忘因子
    % M     - Filter order (number of coefficients) 滤波器阶数(系数)

    if nargin < 3
        lambda = 0.99; % Default forgetting factor 默认遗忘因子
    end
    if nargin < 4
        M = 32; % Default filter order 默认滤波器阶数
    end
    
    n = length(x); % Length of input signal 输入信号长度
    w = zeros(M, 1); % Initialize filter coefficients 初始化滤波器系数
    P = eye(M) * 1000; % Initialize error covariance matrix 初始化误差协方差矩阵
    y = zeros(n, 1); % Output signal 输出信号
    e = zeros(n, 1); % Error signal 误差信号

    for k = M:n
        x_k = flipud(x(k-M+1:k)); % Current input signal frame (length M) 信号分帧
        y(k) = w' * x_k; % Calculate filter output 计算滤波器输出
        e(k) = d(k) - y(k); % Calculate error 计算误差
        
        k_k = (P * x_k) / (lambda + x_k' * P * x_k); % Calculate gain vector 计算增益向量
        w = w + k_k * e(k); % Update filter coefficients 更新滤波器系数
        
        P = (P - k_k * (x_k' * P)) / lambda; % Update error covariance matrix 更新误差协方差矩阵
    end
end