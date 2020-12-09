%[2017]-"A new meta-heuristic butterfly-inspired algorithm"

% (8/12/2020)

function ABO = jArtificialButterflyOptimization(feat,label,opts)
% Parameters
lb     = 0;
ub     = 1; 
thres  = 0.5;
step_e = 0.05;   % control number of sunspot 
ratio  = 0.2;    % control step
type   = 1;      % type 1 or 2

if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'ratio'), ratio = opts.ratio; end 
if isfield(opts,'stepe'), step_e = opts.stepe; end 
if isfield(opts,'ty'), type = opts.ty; end
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
% Fitness 
fit  = zeros(1,N); 
fitG = inf;
for i = 1:N
  fit(i) = fun(feat,label,(X(i,:) > thres),opts);
  % Global update
  if fit(i) < fitG
    fitG = fit(i); 
    Xgb  = X(i,:);
  end
end
% Pre 
Xnew = zeros(N,dim);

curve = zeros(1,max_Iter);
curve(1) = fitG;
t = 2; 
% Iteration
while t <= max_Iter
  % Sort butterfly 
  [fit, idx] = sort(fit,'ascend'); 
  X          = X(idx,:);
  % Proportion of sunspot butterfly decreasing from 0.9 to ratio
  num_sun = round(N * (0.9 - (0.9 - ratio) * (t / max_Iter)));
  % Define a, linearly decrease from 2 to 0
  a       = 2 - 2 * (t / max_Iter);
  % Step update (5)
  step    = 1 - (1 - step_e) * (t / max_Iter);
  % {1} Some butterflies with better fitness: Sunspot butterfly
  for i = 1:num_sun
    % Random select a butterfly k, but not equal to i
    R = randperm(N); R(R == i) = [];
    k = R(1);
    % [Version 1]
    if type == 1
      % Randomly select a dimension
      J  = randi([1,dim]);
      % Random number in [-1,1]
      r1 = -1 + 2 * rand();
      % Position update (1)   
      Xnew(i,:) = X(i,:);
      Xnew(i,J) = X(i,J) + (X(i,J) - X(k,J)) * r1;
    % [Version 2]
    elseif type == 2
      % Distance
      dist = norm(X(k,:) - X(i,:));
      r2   = rand();
      for d = 1:dim
        % Position update (2)
        Xnew(i,d) = X(i,d) + ((X(k,d) - X(i,d)) / dist) * ...
          (ub - lb) * step * r2;
      end
    end
    % Boundary
    XB = Xnew(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
    Xnew(i,:) = XB;
  end
  % Fitness
  for i = 1:num_sun
    % Fitness
    Fnew = fun(feat,label,(Xnew(i,:) > thres),opts);
    % Greedy selection
    if Fnew < fit(i)
      fit(i) = Fnew; 
      X(i,:) = Xnew(i,:);
    end
    % Global update
    if fit(i) < fitG
      fitG = fit(i);
      Xgb  = X(i,:);
    end
  end
  
  % {2} Some butterflies: Canopy butterfly
  for i = num_sun + 1 : N
    % Random select a sunspot butterfly
    k = randi([1,num_sun]);
    % [Version 1]
    if type == 1
      % Randomly select a dimension
      J  = randi([1,dim]); 
      % Random number in [-1,1]
      r1 = -1 + 2 * rand();
      % Position update (1)
      Xnew(i,:) = X(i,:);
      Xnew(i,J) = X(i,J) + (X(i,J) - X(k,J)) * r1;
    % [Version 2]
    elseif type == 2
      % Distance
      dist = norm(X(k,:) - X(i,:));
      r2   = rand();
      for d = 1:dim
        % Position update (2)
        Xnew(i,d) = X(i,d) + ((X(k,d) - X(i,d)) / dist) * ...
          (ub - lb) * step * r2;
      end
    end
    % Boundary
    XB = Xnew(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
    Xnew(i,:) = XB;
  end
  % Fitness
  for i = num_sun + 1 : N
    % Fitness
    Fnew = fun(feat,label,(Xnew(i,:) > thres),opts);
    % Greedy selection
    if Fnew < fit(i)
      fit(i) = Fnew; 
      X(i,:) = Xnew(i,:);
    else
      % Random select a butterfly
      k  = randi([1,N]);
      % Fly to new location
      r3 = rand(); 
      r4 = rand();
      for d = 1:dim
        % Compute D (4)
        Dx     = abs(2 * r3 * X(k,d) - X(i,d));
        % Position update (3)
        X(i,d) = X(k,d) - 2 * a * r4 - a * Dx;
      end
      % Boundary
      XB = X(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb; 
      X(i,:) = XB;
      % Fitness
      fit(i) = fun(feat,label,(X(i,:) > thres),opts);
    end
    % Global update
    if fit(i) < fitG
      fitG = fit(i);
      Xgb  = X(i,:);
    end
  end
  curve(t) = fitG;
  if type == 1
    fprintf('\nIteration %d Best (ABO 1)= %f',t,curve(t))
  elseif type == 2
    fprintf('\nIteration %d Best (ABO 2)= %f',t,curve(t))
  end
  t = t + 1;
end
% Select features
Pos   = 1:dim; 
Sf    = Pos((Xgb > thres) == 1); 
sFeat = feat(:,Sf);
% Store results
ABO.sf = Sf; 
ABO.ff = sFeat;
ABO.nf = length(Sf);
ABO.c  = curve;
ABO.f  = feat;
ABO.l  = label;
end





