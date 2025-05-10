clear

r = 1;
h = 0.1;
k_ = r*h;

x = -2:h:2;
y = -2:k_:2;

u = zeros(numel(x), numel(y));

[X, Y] = meshgrid(x, y);
% Some kind of double bell shape initial value
Z =  X.*exp(-(X-.2).^2-Y.^2);
u(:, :, 1) = Z;
% u(18:22, 18:22)

clear X Y Z;
% D fellow boundary condition for all sides
DBC = 0;
u(:, 1, 1) = DBC;
u(:, end, 1) = DBC;
u(1, :, 1) = DBC;
u(end, :, 1) = DBC;

% Arbitrary value for number of k-iteratiosn
k_iterations = 10000;

for k = 1 : k_iterations
    u(:, 1, k) = ones(size(x));
    % u(:, end, k) = ones(size(x));
    for j = 2 : numel(y)-1
        for i = 2 : numel(x)-1
            u(i, j, k+1) = (1/4) * ...
            (u(i+1,j,k) + u(i-1,j,k) + u(i,j+1,k) + u(i,j-1,k));
        end
    end
    
end


animate3D(x, y, u);



