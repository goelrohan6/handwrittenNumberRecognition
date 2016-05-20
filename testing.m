% load your weight matrix
testInputImage = imread('0.png'); % input the image
testInputImage = testInputImage(:); % convert input image into a column matrix
testInputImage = [1;testInputImage]; % add bias = 1 at first position of matrix
testInputImage = double(testInputImage); % convert matrix to double
testHiddenOutput = [1;sigmf(W1'*testInputImage, [1 0])]; % compute hidden layer output
testPredictedoutput = sigmf(W2'*testHiddenOutput, [1 0]); % predict the output
[max_unit, m] = max(testPredictedoutput);
m-1