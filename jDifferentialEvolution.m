%[1997]-"Differential evolution - A simple and efficient heuristic for
%global optimization over continuous spaces"

% (9/12/2020)

function DE = jDifferentialEvolution(feat,label,opts)
% Parameters
lb    = 0; 
ub    = 1;
thres = 0.5;
CR    = 0.9;  % crossover rate
F     = 0.5;  % constant factor

if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'thres'), thres = opts.thres; end
if isfield(opts,'CR'), CR = opts.CR; end
if isfield(opts,'F'), F = opts.F; end

% Function
fun = @jFitnessFunction; 
% Dimension 
dim = size(feat,2);
% Initialize positions 
% Initial 
X   = zeros(N,dim); 
for i = 1:N
  for d = 1:dim
    X(i,d) = lb + (ub - lb) * rand();
  end
end  
% Fitness
fit  = zeros(1,N); 
fitG = inf; 
for i = 1:N
  fit(i) = fun(feat,label,X(i,:) > thres,opts);
  % Best update
  if fit(i) < fitG
    fitG = fit(i);
    Xgb  = X(i,:);
  end
end
% Pre
U = zeros(N,dim); 
V = zeros(N,dim); 

curve = zeros(1,max_Iter); 
curve(1) = fitG;
t = 2; 
while t <= max_Iter
	for i = 1:N
    % Choose r1, r2, r3 randomly, but not equal to i & each other
    RN = randperm(N); RN(RN == i) = [];
    r1 = RN(1); 
    r2 = RN(2); 
    r3 = RN(3);
    % Mutation (2)
    for d = 1:dim
      V(i,d) = X(r1,d) + F * (X(r2,d) - X(r3,d));
    end
    % Random select a index [1,D]
    rnbr = randi([1,dim]); 
    % Crossover (3-4)
    for d = 1:dim
      if rand() <= CR || d == rnbr 
        U(i,d) = V(i,d);
      else
        U(i,d) = X(i,d);
      end
    end
    % Boundary
    XB = U(i,:); XB(XB > ub) = ub; XB(XB < lb) = ub;
    U(i,:) = XB;
    % Fitness
    Fnew = fun(feat,label,(U(i,:) > thres),opts);
    % Selection
    if Fnew <= fit(i)
      X(i,:) = U(i,:);
      fit(i) = Fnew;
    end
    % Best update
    if fit(i) < fitG
      fitG = fit(i);
      Xgb  = X(i,:);
    end
  end
  curve(t) = fitG;
  fprintf('\nIteration %d Best (DE)= %f',t,fitG)
  t = t + 1;
end
% Select features based on selected index
Pos   = 1:dim; 
Sf    = Pos((Xgb > thres) == 1);
sFeat = feat(:,Sf); 
% Store results
DE.sf = Sf; 
DE.ff = sFeat; 
DE.nf = length(Sf);
DE.c  = curve;
DE.f  = feat;
DE.l  = label;
end





