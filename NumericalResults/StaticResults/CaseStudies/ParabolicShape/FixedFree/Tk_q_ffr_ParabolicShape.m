clc;
clear;
% syms x
% A = [21.0080 -x; -x 6000.0014 + 9.8200*x];
% A = 17.0487*x^2 - 6694.2885*x + 126048.0294 ;
% A = [2002.5427 -x; -x 485713.5839]
% 
% format shortE
% vpa(solve(det(A)==0))
% vpa(solve(A==0))
% [a, b, c] = parabolic(0.095, 0.06125, 0.05, 3)
% error();

syms E nu A SF I P L n a b c h x var q v(x) v_eb(x) G;

assumeAlso(E, 'real');
assumeAlso(G, 'real');
assumeAlso(nu, 'real');
assumeAlso(A, 'real');
assumeAlso(SF, 'real');
assumeAlso(I, 'real');
assumeAlso(P, 'real');
assumeAlso(L, 'real');
assumeAlso(n, 'real');
assumeAlso(a, 'real');
assumeAlso(b, 'real');
assumeAlso(c, 'real');
assumeAlso(h, 'real');
assumeAlso(x, 'real');
assumeAlso(var, 'real');
assumeAlso(q, 'real');
% % 41, 41, 41, 191, 383
clc
% PINN: Import columns as column vectors 
% T =  readtable('Tk_q_ffr_IParabolic_41.csv');
% x2 = T.x;
% Dy = T.Dy;
% Rz = T.Rz;

% FEM2Noded: Import columns as column vectors 
% FEM2 =  readtable('FEM-2Noded/41Nodes/FEM-41Nodes-ffr-q-varsec2_2Noded-nodesresults.csv');
% fem2x2 = FEM2.x;
% fem2Dy = FEM2.Dy;
% fem2Rz = FEM2.Rz;

% % FEM3Noded: Import columns as column vectors 
% FEM3 =  readtable('FEM-3Noded/41Nodes/FEM-41Nodes-ffr-q-varsec2_3Noded-nodesresults.csv');
% fem3x2 = FEM3.x;
% fem3Dy = FEM3.Dy;
% fem3Rz = FEM3.Rz;

qi = 0.25;
qm = 0.50;
qf = 0.20;
L = 2.0; % Length of the beam (m)
[a, b, c] = parabolic(qi, qm, qf, L);
% a = -10/9;
% b = 10/3;
% c = 0.5;


%INPUT DATA
E = 100; % Young Modulus (Pa = N/m^2)
nu = 0.3;
G = E / (2 * (1 + nu));
q = -1.0; % Load
k = 9/10;  % Form factor 
n = 41; % Number of 'node values'
var = L/(n-1); % Division of the beam domain based on the number of nodes
x_var = 0:var:L;

% ====EULER-BERNOULLI=================================

cte = 32 * q / (E * pi); % Disttributed load
f = cte * ((-2*L*x+x^2 + L^2) / (a * x ^ 2 + b * x + c) ^ 4);  %  drot_dx

g = int(f, x);
C1 = subs(g, x, 0);
rot_aux1 = g - C1;   % Rotation of the beam
% h = int(rot_aux1, x);
% C2 = subs(h, x, 0);
% u_aux1 = h - C2;
% 
% v_eb = subs(u_aux1, x, x_var);
% rot_eb = subs(rot_aux1, x, x_var);


% =============TIMOSHENKO=========================
mesh = [5, 11, 17, 23, 29, 35, 41, 200, 300, 500];
X = cell(1, 10);
i = 1;
for m = mesh
   var = L/(m-1);
   x_aux = 0:var:L;
   X{i} = x_aux;
   i = i + 1;
end

I = (pi/64)*(a * x ^ 2 + b * x + c) ^ 4;
A = (pi/4)*(a * x ^ 2 + b * x + c) ^ 2;
Irot_diff = I*f;
eqn = E * diff(Irot_diff, x) + G*A*k*(diff(v,x) - rot_aux1) == 0;
cond = [v(0)==0];
vSol(x) = dsolve(eqn, cond);

i = 1;
fileg = 'Tk_ffr_q_ParabolicShape_ref_circ_';
for m = mesh
    x_var = X{i};
    v = vSol(x_var);
    rot = subs(rot_aux1, x, x_var);
    
    M_aux = [x_var', v', rot'];
    M = double(vpa(M_aux));
    data = table(double(vpa(x_var')), double(vpa(v')), double(vpa(rot')), 'VariableNames', {'x', 'Dy', 'Rz'});
    headers = ['x' 'Dy' 'Rz'];
    writetable(data, strcat(fileg, string(m), '.csv'));
    i = i + 1;
end

% digits(6);
% Nfem2 = norm(fem2Dy - v)/norm(v);
% Nfem3 = norm(fem3Dy - v)/norm(v);
% Npinn = norm(Dy - v)/norm(v);
% fprintf("\nNorm FEM 2-noded : %.4f", vpa(Nfem2));
% fprintf("\nNorm FEM 3-noded : %.4f", vpa(Nfem3));
% fprintf("Norm PINN: %.4f", vpa(Npinn));
% 
% m = figure;
% p = plot(x_var,v ,'k-');
% % p = plot(x_var,v ,'k-', x_var, Dy, 'r-', fem2x2, fem2Dy, 'g-', fem3x2, fem3Dy, 'b-');
% % p = plot(x_var,v ,'k-', x_var,v_eb, 'r');
% % p = plot(x_var,v ,'k-');
% 
% % Setting the thickness of each curve
% p(1).LineWidth = 2;
% % p(4).LineWidth = 2;
% xlim([0 L])
% 
% xlabel('x (m)','FontSize',16);
% ylabel('Displacements (m)','FontSize',16);
% 
% grid
% title(gca, 'Displacements');
% set(gca,'FontSize',16);
% legend({'exact'},'Location','best','FontSize',10);
% set(m,'PaperOrientation','landscape');
% set(m,'PaperUnits','normalized');
% set(m,'PaperPosition', [0 0 1 1]);
% % print(gcf, '-dpdf', 'EB_ffr_IParabolic_100_Dy.pdf');
% 
% % ROTATIONS ========================================================
% t = figure;
% p = plot(x_var,rot ,'k-');
% % p = plot(x_var,rot ,'k-',x_var,rot_eb, 'r');
% % p = plot(x_var,rot ,'k-', x_var, Rz, 'r-', fem2x2, fem2Rz, 'g-', fem3x2, fem3Rz, 'b-');
% % p = plot(x_var,rot,'k-');
% 
% % Setting the thickness of each curve
% p(1).LineWidth = 2;
% xlim([0 L])
% 
% xlabel('x (m)','FontSize',16);
% ylabel('Rotations (rad)','FontSize',16);
% 
% grid
% title(gca, 'Rotations');
% set(gca,'FontSize',16);
% legend({'exact'},'Location','best','FontSize',10);
% set(t,'PaperOrientation','landscape');
% set(t,'PaperUnits','normalized');
% set(t,'PaperPosition', [0 0 1 1]);
% % print(gcf, '-dpdf', 'EB_ffr_IParabolic_100_Rz.pdf');

