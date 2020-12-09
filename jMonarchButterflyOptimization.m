%[2015]-"Monarch butterfly optimization"

% (9/12/2020)

function MBO = jMonarchButterflyOptimization(feat,label,opts)
% Parameters
lb        = 0;
ub        = 1; 
thres     = 0.5; 
peri      = 1.2;     % migration period
p         = 5/12;    % ratio
Smax      = 1;       % maximum step
BAR       = 5/12;    % butterfly adjusting rate
num_land1 = 4;       % number of butterflies in land 1
beta      = 1.5;     % levy component

if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'peri'), peri = opts.peri; end 
if isfield(opts,'p'), p = opts.p; end 
if isfield(opts,'Smax'), Smax = opts.Smax; end 
if isfield(opts,'BAR'), BAR = opts.BAR; end 
if isfield(opts,'beta'), beta = opts.beta; end 
if isfield(opts,'N1'), num_land1 = opts.N1; end 
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
  % Global best update
  if fit(i) < fitG
    fitG = fit(i);
    Xgb  = X(i,:);
  end  
end 
% Pre
Xnew = zeros(N,dim);
Fnew = zeros(1,N);

curve = zeros(1,max_Iter);
curve(1) = fitG;
t = 2;
% Iterations
while t <= max_Iter
  % Sort butterfly
  [fit, idx] = sort(fit,'ascend');
  X          = X(idx,:); 
  % Weight factor (8)
  alpha = Smax / (t ^ 2);
  % {1} First land: Migration operation
  for i = 1:num_land1
    for d = 1:dim
      % Random number (2) 
      r = rand() * peri;
      if r <= p
        % Random select a butterfly in land 1
        r1 = randi([1,num_land1]);
        % Update position (1)
        Xnew(i,d) = X(r1,d);
      else
        % Random select a butterfly in land 2
        r2 = randi([num_land1 + 1, N]);
        % Update position (3)
        Xnew(i,d) = X(r2,d);
      end
    end
    % Boundary
    XB = Xnew(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
    Xnew(i,:) = XB;
  end
  
  % {2} Second land: Butterly adjusting operation
  for i = num_land1 + 1 : N
    % Levy distribution (7)
    dx = jLevyDistribution(beta,dim);
    for d = 1:dim
      if rand() <= p
        % Position update (4) 
        Xnew(i,d) = Xgb(d);
      else
        % Random select a butterfly in land 2
        r3 = randi([num_land1 + 1, N]);
        % Update position (5)
        Xnew(i,d) = X(r3,d);   
        % Butterfly adjusting (6)
        if rand () > BAR
          Xnew(i,d) = Xnew(i,d) + alpha * (dx(d) - 0.5);
        end
      end
    end
    % Boundary
    XB = Xnew(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
    Xnew(i,:) = XB;
  end
  
  % {3} Combine population
  for i = 1:N
    % Fitness
    Fnew(i) = fun(feat,label,(Xnew(i,:) > thres),opts);
    % Global best update
    if Fnew(i) < fitG
      fitG = Fnew(i);
      Xgb  = Xnew(i,:);
    end
  end
  % Merge & Select best N solutions
  XX        = [X; Xnew];
  FF        = [fit, Fnew];
  [FF, idx] = sort(FF,'ascend');
  X         = XX(idx(1:N),:);
  fit       = FF(1:N);
  % Save
  curve(t) = fitG;
  fprintf('\nIteration %d Best (MBO)= %f',t,curve(t))
  t = t + 1;
end
% Select features
Pos   = 1:dim;
Sf    = Pos((Xgb > thres) == 1); 
sFeat = feat(:,Sf);
% Store results
MBO.sf = Sf; 
MBO.ff = sFeat; 
MBO.nf = length(Sf); 
MBO.c  = curve; 
MBO.f  = feat;
MBO.l  = label;
end


%// Levy Flight //
function LF = jLevyDistribution(beta,dim)
% Sigma 
nume  = gamma(1 + beta) * sin(pi * beta / 2);
deno  = gamma((1 + beta) / 2) * beta * 2 ^ ((beta - 1) / 2);
sigma = (nume / deno) ^ (1 / beta); 
% Parameter u & v 
u = randn(1,dim) * sigma; 
v = randn(1,dim);
% Step 
step = u ./ abs(v) .^ (1 / beta); 
LF   = step;
end

