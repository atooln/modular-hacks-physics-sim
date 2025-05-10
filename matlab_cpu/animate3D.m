


function animate3D(x, y, u)
    s1 = surf(x, y, u(:, :, 1));
    zlim([-1, 1])
    sim_i = 1;
    while sim_i < numel(u(1,1,:))
        s1.ZData = u(:, :, sim_i);
        pause(0.05);
        sim_i = sim_i + 5;
    end
end