% Project 2, Three Mass System
% Masroor Khonkhodzhaev
clear


% Spring constants
% k = [2, .1, .1, 2] * 3; % Loosely coupled
% k = [1, 1, 1, 1] * 3;
k = [2, 0.3, 0.1, 0.6] * 3; % Chaotic 

% Time stuff
dt = 0.0001;
T = 5;
t = 0:dt:T;

% Matrix for positions of all three masses
x = zeros([length(t), 3]);

% Interesting settings
x(1, :) = [-.4, 0, .4];
% x(1, :) = [.4, .4, .4];
% x(1, :) = [0, 0, .8];
% x(1, :) = [.4, -.1, .2];

% Matrix for velicities
v = zeros([length(t), 3]);
v(1, :) = [0, 0, 0];

for i = 1 : numel(t) - 1
    % Updating velocities
    v(i+1, 1) = v(i, 1) - ...
        ( k(1)*x(i,1)  +  k(2)*(x(i,1) - x(i,2)) )*dt;

    v(i+1, 2) = v(i, 2) - ...
        ( k(2)*(x(i,2) - x(i,1))  +  k(3)*(x(i,2) - x(i,3)) )*dt;

    v(i+1, 3) = v(i, 3) - ...
        ( k(3)*(x(i,3) - x(i,2))  +  k(4)*x(i,3) )*dt;

    % Updating positions
    x(i+1, 1) = x(i, 1) + v(i+1, 1)*dt;
    x(i+1, 2) = x(i, 2) + v(i+1, 2)*dt;
    x(i+1, 3) = x(i, 3) + v(i+1, 3)*dt;
end

close all;
f1 = figure("Name", "Positions of masses");
plot(t, x);
legend("m1", "m2", "m3");


KE = v.^2./2;
PE(:, 1) = k(1).*x(:, 1).^2./2;
PE(:, 2) = k(2).*(x(:,2) - x(:,1)).^2./2;
PE(:, 3) = k(3).*(x(:,3) - x(:,2)).^2./2;
PE(:, 4) = k(4).*x(:,3).^2./2;

E = sum(KE, 2) + sum(PE, 2);


f2 = figure("Name", "Energies");
plot(t, [sum(KE, 2), sum(PE, 2), E])
legend("Kinetic", "Potential", "Total");

testMotion1 = 1. + x(:, 1);
testMotion2 = 3. + x(:, 2);
testMotion3 = 5. + x(:, 3);

% testMotion1 = 1. + 0.3 * sin(t);
% testMotion2 = 3. + 0.3 * sin(t+10);
% testMotion3 = 6. - testMotion1;

f3 = figure("Name", "Simulation");
% Springs
hold on;
s1 = plot([0, 1], [0.5, 0.5]);
s2 = plot([2, 3], [0.5, 0.5]);
s3 = plot([4, 5], [0.5, 0.5]);
s4 = plot([6, 7], [0.5, 0.5]);
hold off;

% Unmovable sides
rectangle('Position',[-1, 0, 1, 1], 'FaceColor',[.6 .6 .6]);
rectangle('Position',[7, 0, 1, 1], 'FaceColor',[.6 .6 .6]);

% Moving Masses
r1 = rectangle('Position',[testMotion1(1), 0, 1, 1], 'Curvature',0.05);
r2 = rectangle('Position',[testMotion2(1), 0, 1, 1], 'Curvature',0.05);
r3 = rectangle('Position',[testMotion3(1), 0, 1, 1], 'Curvature',0.05);

xlim([-1, 8]);
ylim([-3, 5]);

sim_i = 1;
while sim_i < numel(t)
    % Change mass rectangle positions
    r1.Position = [testMotion1(sim_i), 0, 1, 1];
    r2.Position = [testMotion2(sim_i), 0, 1, 1];
    r3.Position = [testMotion3(sim_i), 0, 1, 1];
    % Change spring endpoints
    s1.XData = [0, r1.Position(1)];
    s2.XData = [r1.Position(1)+1, r2.Position(1)];
    s3.XData = [r2.Position(1)+1, r3.Position(1)];
    s4.XData = [r3.Position(1)+1, 7];
    
    % Simulation params
    pause(0.001);
    sim_i = sim_i + 50;
end

disp("Completed Successfully. Consider donating to our research fund.")