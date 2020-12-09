%[2019]-"Harris hawks optimization: Algorithm and applications"

% (8/12/2020)

function HHO = jHarrisHawksOptimization(feat,label,opts)
% Parameters
lb    = 0;
ub    = 1; 
thres = 0.5; 
beta  = 1.5;   % levy component

if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'N'), N = opts.N; end
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
% Pre
fit  = zeros(1,N);
fitR = inf;
Y    = zeros(1,dim);
Z    = zeros(1,dim); 

curve = inf;
t = 1; 
% Iterations
while t <= max_Iter
  for i = 1:N
    % Fitness
    fit(i) = fun(feat,label,(X(i,:) > thres),opts);
    % Rabbit update
    if fit(i) < fitR
      fitR = fit(i); 
      Xrb  = X(i,:);
    end
  end
  % Mean position of hawk (2)
  X_mu = mean(X,1);
  for i = 1:N
    % Random number in [-1,1]
    E0 = -1 + 2 * rand();
    % Escaping energy of rabbit (3)
    E  = 2 * E0 * (1 - (t / max_Iter)); 
    % Exploration phase
    if abs(E) >= 1
      % Define q in [0,1]
      q = rand(); 
      if q >= 0.5
        % Random select a hawk k
        k  = randi([1,N]);
        r1 = rand();
        r2 = rand();
        for d = 1:dim
          % Position update (1)
          X(i,d) = X(k,d) - r1 * abs(X(k,d) - 2 * r2 * X(i,d));
        end
      elseif q < 0.5    
        r3 = rand(); 
        r4 = rand();
        for d = 1:dim
          % Update Hawk (1)
          X(i,d) = (Xrb(d) - X_mu(d)) - r3 * (lb + r4 * (ub - lb));
        end
      end
    % Exploitation phase 
    elseif abs(E) < 1
      % Jump strength 
      J = 2 * (1 - rand()); 
      r = rand();
      % {1} Soft besiege
      if r >= 0.5 && abs(E) >= 0.5
        for d = 1:dim
          % Delta X (5)
          DX = Xrb(d) - X(i,d);
          % Position update (4)
          X(i,d) = DX - E * abs(J * Xrb(d) - X(i,d));
        end
      % {2} hard besiege
      elseif r >= 0.5 && abs(E) < 0.5
        for d = 1:dim
          % Delta X (5)
          DX = Xrb(d) - X(i,d);
          % Position update (6)
          X(i,d) = Xrb(d) - E * abs(DX); 
        end
      % {3} Soft besiege with progressive rapid dives
      elseif r < 0.5 && abs(E) >= 0.5
        % Levy distribution (9)
        LF = jLevyDistribution(beta,dim); 
        for d = 1:dim
          % Compute Y (7)
          Y(d) = Xrb(d) - E * abs(J * Xrb(d) - X(i,d));
          % Compute Z (8)
          Z(d) = Y(d) + rand() * LF(d); 
        end
        % Boundary
        Y(Y > ub) = ub; Y(Y < lb) = lb;
        Z(Z > ub) = ub; Z(Z < lb) = lb;
        % Fitness
        fitY = fun(feat,label,(Y > thres),opts);
        fitZ = fun(feat,label,(Z > thres),opts);
        % Greedy selection (10)
        if fitY < fit(i)
          fit(i) = fitY; 
          X(i,:) = Y;
        end
        if fitZ < fit(i)
          fit(i) = fitZ; 
          X(i,:) = Z;
        end
      % {4} Hard besiege with progressive rapid dives
      elseif r < 0.5 && abs(E) < 0.5
        % Levy distribution (9)
        LF = jLevyDistribution(beta,dim); 
        for d = 1:dim
          % Compute Y (12)
          Y(d) = Xrb(d) - E * abs(J * Xrb(d) - X_mu(d));
          % Compute Z (13)
          Z(d) = Y(d) + rand() * LF(d);
        end
        % Boundary
        Y(Y > ub) = ub; Y(Y < lb) = lb;
        Z(Z > ub) = ub; Z(Z < lb) = lb;
        % Fitness
        fitY = fun(feat,label,(Y > thres),opts);
        fitZ = fun(feat,label,(Z > thres),opts);
        % Greedy selection (11)
        if fitY < fit(i)
          fit(i) = fitY;
          X(i,:) = Y;
        end
        if fitZ < fit(i)
          fit(i) = fitZ;
          X(i,:) = Z;
        end        
      end
    end
    % Boundary 
    XB = X(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb; 
    X(i,:) = XB;
  end
  % Save
  curve(t) = fitR;
  fprintf('\nIteration %d Best (HHO)= %f',t,curve(t))
  t = t + 1;
end
% Select features
Pos   = 1:dim;
Sf    = Pos((Xrb > thres) == 1);
sFeat = feat(:,Sf);
% Store results
HHO.sf = Sf;
HHO.ff = sFeat;
HHO.nf = length(Sf);
HHO.c  = curve;
HHO.f  = feat;
HHO.l  = label;
end


%// Levy Flight (9)
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
LF   = 0.01 * step;
end


