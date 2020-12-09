%[2006]-"Ant Colony Optimization"

% (9/12/2020)

function ACS = jAntColonySystem(feat,label,opts)
% Parameters
tau   = 1;      % pheromone value
eta   = 1;      % heuristic desirability
alpha = 1;      % control pheromone
beta  = 1;      % control heuristic
rho   = 0.2;    % pheromone trail decay coefficient
phi   = 0.5;    % pheromena coefficient

if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'tau'), tau = opts.tau; end  
if isfield(opts,'alpha'), alpha = opts.alpha; end 
if isfield(opts,'beta'), beta = opts.beta; end 
if isfield(opts,'rho'), rho = opts.rho; end 
if isfield(opts,'eta'), eta = opts.eta; end  
if isfield(opts,'phi'), phi = opts.phi; end 

% Objective function
fun = @jFitnessFunction; 
% Number of dimensions
dim = size(feat,2); 
% Initial Tau & Eta 
tau = tau * ones(dim,dim); 
eta = eta * ones(dim,dim);
% Pre
fitG = inf; 
fit  = zeros(1,N);
tau0 = tau;

curve = inf; 
t = 1; 
% Iterations
while t <= max_Iter
	% Reset ant
	X = zeros(N,dim); 
	for i=1:N
    % Set number of features
    num_feat = randi([1,dim]);
    % Ant start with random position
    X(i,1)   = randi([1,dim]); 
    k        = [];
    if num_feat > 1
      for d = 2:num_feat
        % Start with previous tour
        k      = [k(1:end), X(i, d-1)];
        % Edge / Probability Selection (4)
        P      = (tau(k(end),:) .^ alpha) .* (eta(k(end),:) .^ beta); 
        % Set selected position = 0 probability (4)
        P(k)   = 0; 
        % Convert probability (4)
        prob   = P ./ sum(P(:)); 
        % Roulette Wheel selection
        route  = jRouletteWheelSelection(prob);
        % Store selected position to be next tour
        X(i,d) = route;
      end
    end
  end
  % Binary
  X_bin = zeros(N,dim);
  for i = 1:N
    % Binary form
    ind           = X(i,:); 
    ind(ind == 0) = [];
    X_bin(i, ind) = 1;
  end
  % Binary version
  for i = 1:N
    % Fitness
    fit(i) = fun(feat,label,X_bin(i,:),opts);
    % Global update
    if fit(i) < fitG
      Xgb  = X(i,:);
      fitG = fit(i); 
    end
  end
  % Tau update 
  tour            = Xgb; 
  tour(tour == 0) = []; 
  tour            = [tour(1:end), tour(1)];
  for d = 1 : length(tour) - 1
    % Feature selected
    x = tour(d);
    y = tour(d + 1);
    % Delta tau
    Dtau = 1 / fitG;
    % Update tau (10)
    tau(x,y) = (1 - phi) * tau(x,y) + phi * Dtau; 
  end
  % Evaporate pheromone (9)
  tau = (1 - rho) * tau + rho * tau0;
  % Save
  curve(t) = fitG;
  fprintf('\nIteration %d Best (ACS)= %f',t,curve(t))
  t = t + 1;
end
% Select features based on selected index
Sf = unique(Xgb);
Sf(Sf == 0) = [];
sFeat = feat(:,Sf); 
% Store results
ACS.sf = Sf;
ACS.ff = sFeat;
ACS.nf = length(Sf);
ACS.c  = curve; 
ACS.f  = feat;
ACS.l  = label;
end
    

%// Roulette Wheel Selection //
function Index = jRouletteWheelSelection(prob)
% Cummulative summation
C = cumsum(prob);
% Random one value, most probability value [0~1]
P = rand();
% Route wheel
for i = 1:length(C)
	if C(i) > P
    Index = i;
    break;
  end
end
end      

