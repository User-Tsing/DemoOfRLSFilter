% Example usage of RLS algorithm
fs = 1000; % Sampling frequency
t = 0:1/fs:1-1/fs; % Time vector
signal_freq = 5; % Signal frequency
xn = sin(2 * pi * signal_freq * t); % Original signal
noise = 0.5 * randn(size(t)); % Noise
dn = xn + noise; % Noisy signal

% Call RLS algorithm
[w, yn, en] = RLS(dn', xn', 0.99, 32); % Transpose to match MATLAB format

% Plot results
figure;
subplot(4,2,1);
plot(xn);
title("Original Signal");
subplot(4,2,2);
xf=abs(fft(xn));
plot(xf);
title("fft:Original Signal");
subplot(4,2,3);
plot(noise);
title("Noise");
subplot(4,2,4);
noif=abs(fft(noise));
plot(noif);
title("fft:Noise");
subplot(4,2,5);
plot(dn, 'DisplayName', 'Noisy Signal', 'Color', 'r', 'LineWidth', 0.5);
title("Noisy signal");
subplot(4,2,6);
df=abs(fft(dn));
plot(df);
title("fft:Noisy signal");
subplot(4,2,7);
plot(yn, 'DisplayName', 'Filtered Signal', 'Color', 'b', 'LineWidth', 1);
title("Filtered Signal");
subplot(4,2,8);
yf=abs(fft(yn));
plot(yf);
title("fft:Filtered Signal");
legend show;
grid on;