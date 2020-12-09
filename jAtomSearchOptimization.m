%[2019]-"Atom search optimization and its application to solve a
%hydrogeologic parameter estimation problem"

% (8/12/2020)

function ASO = jAtomSearchOptimization(feat,label,opts)
% Parameters
lb    = 0;
ub    = 1; 
thres = 0.5; 
alpha = 50;    % depth weight
beta  = 0.2;   % multiplier weight

if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'alpha'), alpha = opts.alpha; end 
if isfield(opts,'beta'), beta = opts.beta; end 
if isfield(opts,'thres'), thres = opts.thres; end 

% Objective function
fun = @jFitnessFunction; 
% Number of dimensions
dim = size(feat,2); 
% Initial 
X   = zeros(N,dim);
for i = 1:N
	for d = 1:dim
    X(i,d) = lb + (ub - lb) * rand(); 
  end
end
V   = zeros(N,dim); 
for i = 1:N
	for d = 1:dim
    V(i,d) = lb + (ub - lb) * rand();
  end
end
% Pre
temp_A = zeros(N,dim); 
fitG   = inf; 
fit    = zeros(1,N);

curve = inf;
t = 1;
% Iteration
while t <= max_Iter
  for i = 1:N
    % Fitness
    fit(i) = fun(feat,label,(X(i,:) > thres),opts);
    % Best update
    if fit(i) < fitG
      fitG  = fit(i);
      Xbest = X(i,:);
    end
  end
  % Best & worst fitness (28-29)
  fitB = min(fit);
  fitW = max(fit); 
  % Number of K neighbor (32)
  Kbest = ceil(N - (N - 2) * sqrt(t / max_Iter));   
  % Mass (26)
  M = exp(-(fit - fitB) ./ (fitW - fitB));
  % Normalized mass (27)
  M = M ./ sum(M);
  % Sort normalized mass in descending order
  [~, idx_M] = sort(M,'descend');
  % Contsraint force (23-24)
  G = exp(-20 * t / max_Iter); 
  E = zeros(N,dim); 
  for i = 1:N      
    XK(1,:) = sum(X(idx_M(1:Kbest),:),1) / Kbest;
    % Length scale (17)
    scale_dist = norm(X(i,:) - XK(1,:),2);   
    for ii = 1:Kbest
      % Select neighbor with higher mass 
      j  = idx_M(ii);
      % Get LJ-potential
      Po = jLJPotential(X(i,:),X(j,:),t,max_Iter,scale_dist);
      % Distance
      dist = norm(X(i,:) - X(j,:),2);
      for d = 1:dim
        % Update (25)
        E(i,d) = E(i,d) + rand() * Po * ((X(j,d) - X(i,d)) / ...
          (dist + eps));
      end
    end
    for d = 1:dim
      E(i,d) = alpha * E(i,d) + beta * (Xbest(d) - X(i,d));
      % Calculate part of acceleration (25)
      temp_A(i,d) = E(i,d) / M(i);
    end
  end
  % Update
  for i = 1:N
    for d = 1:dim
      % Acceleration (25)
      Acce   = temp_A(i,d) * G;
      % Velocity update (30)
      V(i,d) = rand() * V(i,d) + Acce;
      % Position update (31)
      X(i,d) = X(i,d) + V(i,d);
    end
    % Boundary 
    XB = X(i,:); XB(XB > ub) = ub; XB(XB <lb) = lb;
    X(i,:) = XB;
  end
  curve(t) = fitG; 
  fprintf('\nIteration %d Best (ASO)= %f',t,curve(t))
  t = t + 1;
end
% Select features based on selected index
Pos   = 1:dim;
Sf    = Pos((Xbest > thres) == 1);
sFeat = feat(:,Sf); 
% Store results
ASO.sf = Sf; 
ASO.ff = sFeat; 
ASO.nf = length(Sf); 
ASO.c  = curve; 
ASO.f  = feat; 
ASO.l  = label;
end


%// LJ-Potential //
function Potential = jLJPotential(X1,X2,t,max_Iter,scale_dist)
% Calculate LJ-potential
h0 = 1.1; 
u  = 1.24; 
% Equilibration distance [Assume 1.12*(17)~=(17)] 
r  = norm(X1 - X2,2);
% Depth function (15)
n  = (1 - (t - 1) / max_Iter) .^ 3;
% Drift factor (19)
g  = 0.1 * sin((pi / 2) * (t / max_Iter));
% Hmax & Hmin (18)
Hmin = h0 + g; 
Hmax = u;
% Compute H (16)
if r / scale_dist < Hmin
  H = Hmin;
elseif r / scale_dist > Hmax
  H = Hmax;  
else
  H = r / scale_dist;
end           
% Revised version (14,25)
Potential = n * (12 * (-H) ^ (-13) - 6 * (-H) ^ (-7)); 
end

