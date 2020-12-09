%[1995]-"Particle Swarm Optimization" 
%[1998]-"A modified particle swarm optimizer"

% (9/12/2020)

function PSO = jParticleSwarmOptimization(feat,label,opts)
% Parameters
lb    = 0; 
ub    = 1;
thres = 0.5;
c1    = 2;              % cognitive factor
c2    = 2;              % social factor 
w     = 0.9;            % inertia weight
Vmax  = (ub - lb) / 2;  % Maximum velocity 

if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'c1'), c1 = opts.c1; end 
if isfield(opts,'c2'), c2 = opts.c2; end 
if isfield(opts,'w'), w = opts.w; end 
if isfield(opts,'Vmax'), Vmax = opts.Vmax; end 
if isfield(opts,'thres'), thres = opts.thres; end

% Objective function
fun = @jFitnessFunction; 
% Number of dimensions
dim = size(feat,2); 
% Initial 
X   = zeros(N,dim); 
V   = zeros(N,dim); 
for i = 1:N
  for d = 1:dim
    X(i,d) = lb + (ub - lb) * rand();
  end
end  
% Fitness
fit  = zeros(1,N); 
fitG = inf;
for i = 1:N 
  fit(i) = fun(feat,label,(X(i,:) > thres),opts); 
  % Gbest update
  if fit(i) < fitG
    Xgb  = X(i,:); 
    fitG = fit(i);
  end
end
% PBest
Xpb  = X; 
fitP = fit;
% Pre
curve = zeros(1,max_Iter);
curve(1) = fitG;
t = 2;  
% Iterations
while t <= max_Iter
  for i = 1:N
    for d = 1:dim
      r1 = rand();
      r2 = rand();
      % Velocity update (2a)
      VB = w * V(i,d) + c1 * r1 * (Xpb(i,d) - X(i,d)) + ...
        c2 * r2 * (Xgb(d) - X(i,d));
      % Velocity limit
      VB(VB > Vmax) = Vmax;  VB(VB < -Vmax) = -Vmax;
      V(i,d) = VB;
      % Position update (2b)
      X(i,d) = X(i,d) + V(i,d);
    end
    % Boundary
    XB = X(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
    X(i,:) = XB;
    % Fitness
    fit(i) = fun(feat,label,(X(i,:) > thres),opts);
    % Pbest update
    if fit(i) < fitP(i)
      Xpb(i,:) = X(i,:); 
      fitP(i)  = fit(i);
    end
    % Gbest update
    if fitP(i) < fitG
      Xgb  = Xpb(i,:);
      fitG = fitP(i);
    end
  end
  curve(t) = fitG; 
  fprintf('\nIteration %d Best (PSO)= %f',t,curve(t))
  t = t + 1;
end
% Select features based on selected index
Pos   = 1:dim;
Sf    = Pos((Xgb > thres) == 1); 
sFeat = feat(:,Sf); 
% Store results
PSO.sf = Sf; 
PSO.ff = sFeat;
PSO.nf = length(Sf);
PSO.c  = curve;
PSO.f  = feat;
PSO.l  = label;
end



