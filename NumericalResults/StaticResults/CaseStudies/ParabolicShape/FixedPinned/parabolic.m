function [a, b, c] = parabolic(qi, qm, qf, L)

a = (-4*qm + 2*qf + 2*qi)/L^2;
b = (4*qm - qf - 3*qi)/L;
c = qi;

end