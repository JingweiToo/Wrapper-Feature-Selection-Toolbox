%[2020]-"Marine Predators Algorithm: A nature-inspired metaheuristic"

% (8/12/2020)

function MPA = jMarinePredatorsAlgorithm(feat,label,opts)
% Parameters
lb    = 0;
ub    = 1; 
thres = 0.5; 
beta  = 1.5;   % levy component
P     = 0.5;   % constant
FADs  = 0.2;   % fish aggregating devices effect

if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'thres'), thres = opts.thres; end
if isfield(opts,'P'), P = opts.P; end
if isfield(opts,'FADs'), FADs = opts.FADs; end

% Objective function
fun = @jFitnessFunction; 
% Number of dimensions
dim = size(feat,2); 
% Initial (9)
X   = zeros(N,dim);  
for i = 1:N
	for d = 1:dim
    X(i,d) = lb + (ub - lb) * rand(); 
  end
end
% Pre
fit  = zeros(1,N);
fitG = inf;

curve = inf; 
t = 1; 
% Iteration
while t <= max_Iter
  % Fitness
  for i=1:N
    fit(i) = fun(feat,label,(X(i,:) > thres),opts);
    % Best
    if fit(i) < fitG
      fitG = fit(i); 
      Xgb  = X(i,:);
    end
  end
  % Memory saving
  if t == 1
    fitM = fit; 
    Xmb  = X;
  end
  for i = 1:N
    if fitM(i) < fit(i)
      fit(i) = fitM(i);
      X(i,:) = Xmb(i,:);
    end
  end
  Xmb  = X;
  fitM = fit;
  % Construct elite (10)
  Xe   = repmat(Xgb,[N 1]);
  % Adaptive parameter (14)
  CF   = (1 - (t / max_Iter)) ^ (2 * (t / max_Iter));
  % [First phase] (12)
  if t <= max_Iter / 3
    for i = 1:N
      % Brownian random number
      RB = randn(1,dim);  
      for d = 1:dim
        R        = rand();
        stepsize = RB(d) * (Xe(i,d) - RB(d) * X(i,d));
        X(i,d)   = X(i,d) + P * R * stepsize;
      end
      % Boundary
      XB = X(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb; 
      X(i,:) = XB;
    end
  % [Second phase] (13-14)
  elseif t > max_Iter / 3 && t <= 2 * max_Iter / 3
    for i = 1:N
      % First half update (13)
      if i <= N / 2
        % Levy random number
        RL = 0.05 * jLevy(beta,dim);
        for d = 1:dim
          R        = rand();
          stepsize = RL(d) * (Xe(i,d) - RL(d) * X(i,d));
          X(i,d)   = X(i,d) + P * R * stepsize;
        end
      % Another half update (14)
      else
        % Brownian random number
        RB = randn(1,dim); 
        for d = 1:dim
          stepsize = RB(d) * (RB(d) * Xe(i,d) - X(i,d));
          X(i,d)   = Xe(i,d) + P * CF * stepsize;
        end
      end
      % Boundary
      XB = X(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb; 
      X(i,:) = XB;
    end
  % [Third phase] (15)
  elseif t > 2 * max_Iter / 3
    for i = 1:N
      % Levy random number
      RL = 0.05 * jLevy(beta,dim);
      for d = 1:dim
        stepsize = RL(d) * (RL(d) * Xe(i,d) - X(i,d));
        X(i,d)   = Xe(i,d) + P * CF * stepsize;
      end
      % Boundary
      XB = X(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb; 
      X(i,:) = XB;
    end
  end 
  % Fitness
  for i = 1:N
    fit(i) = fun(feat,label,(X(i,:) > thres),opts);
    % Best
    if fit(i) < fitG
      fitG = fit(i);
      Xgb  = X(i,:);
    end
  end
  % Memory saving
  for i = 1:N
    if fitM(i) < fit(i)
      fit(i) = fitM(i); 
      X(i,:) = Xmb(i,:);
    end
  end
  Xmb  = X;
  fitM = fit;
  % Eddy formation and FADs effect (16)
  if rand() <= FADs
    for i = 1:N
      % Compute U
      U = rand(1,dim) < FADs;
      for d = 1:dim
        R      = rand();
        X(i,d) = X(i,d) + CF * (lb + R * (ub - lb)) * U(d);
      end
      % Boundary
      XB = X(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb; 
      X(i,:) = XB;
    end
  else
    % Uniform random number [0,1]
    r   = rand(); 
    % Define two prey randomly
    Xr1 = X(randperm(N),:);
    Xr2 = X(randperm(N),:);
    for i = 1:N   
      for d = 1:dim
        X(i,d) = X(i,d) + (FADs * (1 - r) + r ) * ...
          (Xr1(i,d) - Xr2(i,d));
      end
      % Boundary
      XB = X(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
      X(i,:) = XB;
    end
  end
  % Save
  curve(t) = fitG;
  fprintf('\nIteration %d Best (MPA)= %f',t,curve(t))
  t = t + 1;
end
% Select features based on selected index
Pos   = 1:dim;
Sf    = Pos((Xgb > thres) == 1); 
sFeat = feat(:,Sf); 
% Store results
MPA.sf = Sf; 
MPA.ff = sFeat; 
MPA.nf = length(Sf); 
MPA.c  = curve; 
MPA.f  = feat;
MPA.l  = label;
end


% Levy distribution
function LF = jLevy(beta,dim)
num   = gamma(1 + beta) * sin(pi * beta / 2); 
deno  = gamma((1 + beta) / 2) * beta * 2 ^ ((beta - 1) / 2); 
sigma = (num / deno) ^ (1 / beta);
u     = random('Normal',0,sigma,1,dim);
v     = random('Normal',0,1,1,dim);
LF    = u ./ (abs(v) .^ (1 / beta));
end

