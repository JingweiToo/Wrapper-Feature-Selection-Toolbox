% K-nearest Neighbor (9/12/2020)

function Acc = jknn(feat,label,opts)
% Default of k-value
k = 5;

if isfield(opts,'k'), k = opts.k; end
if isfield(opts,'Model'), Model = opts.Model; end

% Define training & validation sets
trainIdx = Model.training;    testIdx = Model.test;
xtrain   = feat(trainIdx,:);  ytrain  = label(trainIdx);
xvalid   = feat(testIdx,:);   yvalid  = label(testIdx);
% Training model
My_Model = fitcknn(xtrain,ytrain,'NumNeighbors',k); 
% Prediction
pred     = predict(My_Model,xvalid);
% Accuracy
Acc      = sum(pred == yvalid) / length(yvalid);

fprintf('\n Accuracy: %g %%',100 * Acc);
end


