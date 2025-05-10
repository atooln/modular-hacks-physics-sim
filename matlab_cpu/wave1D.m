clear

r = 1;
h = 0.01;
k = r * h;

x = 0:h:1;
t = 0:k:2;

u = zeros([numel(t), numel(x)]);

% initial positions
xSin = 0:pi/21:pi;
u(1, 40:61) = sin(xSin);
% u(1, :) = sin(pi*x) + sin(2*pi*x);

% initial velocity function
phi = zeros([1, numel(x)]);
% phiX = (0:pi/21:pi);
% phi(40:61) = -13*cos(phiX);

% plot(u(1, :))

for i = 2:numel(x) - 1
    u(2, i) = 0.5 * (r^2*u(1, i+1) + ...
        2*(1 - r^2)*u(1, i) + r^2*u(1, i-1)) + k*phi(i);
end

for j = 2:numel(t)-1
    for i = 2:numel(x) - 1
        u(j+1, i) = r^2*u(j, i+1) + 2*(1 - r^2)*u(j, i) + r^2*u(j, i-1) - u(j-1, i);
    end
    % Neumann Boundary Condition
    % u(j+1, 1) = u(j+1, 2);
    % u(j+1, end) = u(j+1, end-1);

    % Absorbing B C
    u(j+1, 1) = u(j, 2) + ((r-1)/(r+1)) * (u(j+1, 2) - u(j, 1));
    u(j+1, end) = u(j, end-1) + ((r-1)/(r+1)) * (u(j+1, end-1) - u(j, end));
end

% All time slices at once in 3D
surf(x, t, u);

% Simulation over time
p1 = plot(x, u(1, :));
xlim([-0.2, 1.2])
ylim([-2.2, 2.2]);

for i = 1:numel(t)
    p1.YData = u(i, :);
    pause(0.1);
end

