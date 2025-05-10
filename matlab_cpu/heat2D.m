clear

r = 1/4;
h = 0.05;
tau = r * h^2;

x = (-1:h:1);
y = (-1:h:1);
t = (0:tau:1);

% 3D Matrix
u = zeros(numel(x), numel(y), numel(t));

% 3D Bell curve as the initial condition
[X, Y] = meshgrid(x, y);
Z = exp(-((X.*3).^2 + (Y.*3).^2));
% Setting the initial condition
u(:, :, 1) = Z;
clear X Y Z;

% D fellow boundary condition
u(:, 1, 1) = 0;
u(:, end, 1) = 0;
u(1, :, 1) = 0;
u(end, :, 1) = 0;

% surf(x, y, u(:, :, 1))

for k = 1 : numel(t)
    for j = 2 : numel(y) - 1
        for i = 2 : numel(x) - 1
            u(i, j, k+1) = (1 - 4*r)*u(i,j,k) + ...
                r*u(i-1,j,k) + r*u(i+1,j,k) + r*u(i,j-1,k) + r*u(i,j+1,k);
        end
    end
    % Neumann Boundary Condition
    % Sides
    u(:, 1, k+1) = u(:, 2, k+1);
    u(:, end, k+1) = u(:, end-1, k+1);
    u(1, :, k+1) = u(2, :, k+1);
    u(end, :, k+1) = u(end-1, :, k+1);
    % Corners
    u(1, 1, k+1) = ( u(1, 2, k+1) + u(2, 1, k+1) ) / 2;
    u(1, end, k+1) = ( u(1, end-1, k+1) + u(2, end, k+1) ) / 2;
    u(end, 1, k+1) = ( u(end-1, 1, k+1) + u(end, 2, k+1) ) / 2;
    u(end, end, k+1) = ( u(end-1, end, k+1) + u(end, end-1, k+1) ) / 2;
end


% animate3D(x, y, u);

s1 = surf(x, y, u(:, :, 1));
clim([0, 1])
colorbar
zlim([0, 1])
sim_i = 1;
while sim_i < numel(t)
    s1.ZData = u(:, :, sim_i);
    pause(0.1);
    sim_i = sim_i + 5;
end