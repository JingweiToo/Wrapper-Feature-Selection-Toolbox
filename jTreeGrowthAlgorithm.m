%[2018]-"Tree growth algorithm (TGA): A novel approach for solving
%optimization problems"

function TGA = jTreeGrowthAlgorithm(feat,label,opts)
% Parameters
lb         = 0;
ub         = 1; 
thres      = 0.5; 
num_tree1  = 3;    % size of first group
num_tree2  = 5;    % size of second group
num_tree4  = 3;    % size of fourth group
theta      = 0.8;  % tree reduction rate of power
lambda     = 0.5;  % control nearest tree

if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'N1'), num_tree1 = opts.N1; end 
if isfield(opts,'N2'), num_tree2 = opts.N2; end 
if isfield(opts,'N4'), num_tree4 = opts.N4; end 
if isfield(opts,'theta'), theta = opts.theta; end 
if isfield(opts,'lambda'), lambda = opts.lambda; end 
if isfield(opts,'thres'), thres = opts.thres; end 

% Limit number of N4 to N1
if num_tree4 > num_tree1 + num_tree2
  num_tree4 = num_tree1 + num_tree2; 
end
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
  % Best 
  if fit(i) < fitG
    fitG = fit(i); 
    Xgb  = X(i,:);
  end
end
% Sort tree from best to worst
[fit, idx] = sort(fit,'ascend');
X          = X(idx,:); 
% Initial
dist = zeros(1,num_tree1 + num_tree2);
X1   = zeros(num_tree1,dim);
Xnew = zeros(num_tree4,dim);
Fnew = zeros(1,num_tree4);

curve = zeros(1,max_Iter);
curve(1) = fitG;
t = 2;
% Iterations
while t <= max_Iter
	% {1} Best trees group
  for i = 1:num_tree1
    r1 = rand();
    for d = 1:dim
      % Local search (1)
      X1(i,d) = (X(i,d) / theta) + r1 * X(i,d);
    end
  	% Boundary
    XB = X1(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
    X1(i,:) = XB;
    % Fitness
    fitT = fun(feat,label,(X1(i,:) > thres),opts);
    % Greedy selection
    if fitT <= fit(i)
      X(i,:) = X1(i,:);
      fit(i) = fitT;
    end
  end
  % {2} Competitive for light tree group
  X_ori = X;
  for i = num_tree1 + 1 : num_tree1 + num_tree2
    % Neighbor tree
    for j = 1 : num_tree1 + num_tree2           
      if j ~= i
        % Compute Euclidean distance (2)
        dist(j) = sqrt(sum((X_ori(j,:) - X_ori(i,:)) .^ 2));
      else
        % Solve same tree problem
        dist(j) = inf;
      end
    end
    % Find 2 trees with shorter distance
    [~, idx] = sort(dist,'ascend'); 
    T1       = X_ori(idx(1),:);
    T2       = X_ori(idx(2),:); 
    % Alpha in [0,1]
    alpha    = rand();
    for d = 1:dim
      % Compute linear combination between 2 shorter tree (3)
      y = lambda * T1(d) + (1 - lambda) * T2(d);
      % Move tree i between 2 adjacent trees (4)
      X(i,d) = X(i,d) + alpha * y;
    end
    % Boundary
    XB = X(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
    X(i,:) = XB;
    % Fitness
    fit(i) = fun(feat,label,(X(i,:) > thres),opts);
  end
  % {3} Remove and replace group
  for i = num_tree1 + num_tree2 + 1 : N
    for d = 1:dim
      % Generate new tree by remove worst tree
      X(i,d) = lb + (ub - lb) * rand();
    end
    % Fitness
    fit(i) = fun(feat,label,(X(i,:) > thres),opts);
  end
  % {4} Reproduction group
  for i = 1:num_tree4
    % Random a best tree
    r     = randi([1,num_tree1]);
    Xbest = X(r,:);
    % Mask operator
    mask  = randi([0,1],1,dim);
    % Mask opration between new & best trees
    for d = 1:dim
      % Generate new solution 
      Xn = lb + (ub - lb) * rand();
      if mask(d) == 1
        Xnew(i,d) = Xbest(d);
      elseif mask(d) == 0
        % Generate new tree
        Xnew(i,d) = Xn;
      end
    end
    % Fitness
    Fnew(i) = fun(feat,label,(Xnew(i,:) > thres),opts);
  end
  % Sort population get best nPop trees
  XX        = [X; Xnew];
  FF        = [fit, Fnew];
  [FF, idx] = sort(FF,'ascend');
  X         = XX(idx(1:N),:);
  fit       = FF(1:N);
  % Global best
  if fit(1) < fitG
    fitG = fit(1); 
    Xgb  = X(1,:);
  end
  curve(t) = fitG;
  fprintf('\nIteration %d Best (TGA)= %f',t,curve(t))
  t = t + 1;
end
% Select features
Pos   = 1:dim;
Sf    = Pos((Xgb > thres) == 1); 
sFeat = feat(:,Sf);
% Store results
TGA.sf = Sf; 
TGA.ff = sFeat; 
TGA.nf = length(Sf); 
TGA.c  = curve;
TGA.f  = feat;
TGA.l  = label;
end





