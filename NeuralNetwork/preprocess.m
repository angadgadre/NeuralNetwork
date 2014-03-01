function [train_data, train_label, validation_data, ...
    validation_label, test_data, test_label] = preprocess()
% preprocess function loads the original data set, performs some preprocess
%   tasks, and output the preprocessed train, validation and test data.

% Input:
% Although this function doesn't have any input, you are required to load
% the MNIST data set from file 'mnist_all.mat'.

% Output:
% train_data: matrix of training set. Each row of train_data contains 
%   feature vector of a image
% train_label: vector of label corresponding to each image in the training
%   set
% validation_data: matrix of training set. Each row of validation_data 
%   contains feature vector of a image
% validation_label: vector of label corresponding to each image in the 
%   training set
% test_data: matrix of training set. Each row of test_data contains 
%   feature vector of a image
% test_label: vector of label corresponding to each image in the testing
%   set

% Some suggestions for preprocessing step:
% - divide the original data set to training, validation and testing set
%       with corresponding labels
% - convert original data set from integer to double by using double()
%       function
% - normalize the data to [0, 1]
% - feature selection

load('mnist_all.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% TODO : Feature Selection Code Here %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sumtrain0 = sum(train0);
sumtrain1 = sum(train1);
sumtrain2 = sum(train2);
sumtrain3 = sum(train3);
sumtrain4 = sum(train4);
sumtrain5 = sum(train5);
sumtrain6 = sum(train6);
sumtrain7 = sum(train7);
sumtrain8 = sum(train8);
sumtrain9 = sum(train9);

sumtrain10 = [sumtrain0; sumtrain1; sumtrain2; sumtrain3; sumtrain4; sumtrain5; sumtrain6; sumtrain7; sumtrain8; sumtrain9];
total = sum(sumtrain10);
[~,c] = find(total > 0);
indexCell = num2cell(c,1);
indexCell2 = num2cell(ones(1, size(c,2)),1);

train0 = train0(:, sub2ind([1 size(train0, 2)], [indexCell2{:}], [indexCell{:}]));
train1 = train1(:, sub2ind([1 size(train1, 2)], [indexCell2{:}], [indexCell{:}]));
train2 = train2(:, sub2ind([1 size(train2, 2)], [indexCell2{:}], [indexCell{:}]));
train3 = train3(:, sub2ind([1 size(train3, 2)], [indexCell2{:}], [indexCell{:}]));
train4 = train4(:, sub2ind([1 size(train4, 2)], [indexCell2{:}], [indexCell{:}]));
train5 = train5(:, sub2ind([1 size(train5, 2)], [indexCell2{:}], [indexCell{:}]));
train6 = train6(:, sub2ind([1 size(train6, 2)], [indexCell2{:}], [indexCell{:}]));
train7 = train7(:, sub2ind([1 size(train7, 2)], [indexCell2{:}], [indexCell{:}]));
train8 = train8(:, sub2ind([1 size(train8, 2)], [indexCell2{:}], [indexCell{:}]));
train9 = train9(:, sub2ind([1 size(train9, 2)], [indexCell2{:}], [indexCell{:}]));


% sumtest0 = sum(test0);
% sumtest1 = sum(test1);
% sumtest2 = sum(test2);
% sumtest3 = sum(test3);
% sumtest4 = sum(test4);
% sumtest5 = sum(test5);
% sumtest6 = sum(test6);
% sumtest7 = sum(test7);
% sumtest8 = sum(test8);
% sumtest9 = sum(test9);
% 
% sumtest10 = [sumtest0; sumtest1; sumtest2; sumtest3; sumtest4; sumtest5; sumtest6; sumtest7; sumtest8; sumtest9];
% total = sum(sumtest10);
% [~,c] = find(total >= 6000);
% indexCell = num2cell(c,1);
% indexCell2 = num2cell(ones(1, size(c,2)),1);

test0 = test0(:, sub2ind([1 size(test0, 2)], [indexCell2{:}], [indexCell{:}]));
test1 = test1(:, sub2ind([1 size(test1, 2)], [indexCell2{:}], [indexCell{:}]));
test2 = test2(:, sub2ind([1 size(test2, 2)], [indexCell2{:}], [indexCell{:}]));
test3 = test3(:, sub2ind([1 size(test3, 2)], [indexCell2{:}], [indexCell{:}]));
test4 = test4(:, sub2ind([1 size(test4, 2)], [indexCell2{:}], [indexCell{:}]));
test5 = test5(:, sub2ind([1 size(test5, 2)], [indexCell2{:}], [indexCell{:}]));
test6 = test6(:, sub2ind([1 size(test6, 2)], [indexCell2{:}], [indexCell{:}]));
test7 = test7(:, sub2ind([1 size(test7, 2)], [indexCell2{:}], [indexCell{:}]));
test8 = test8(:, sub2ind([1 size(test8, 2)], [indexCell2{:}], [indexCell{:}]));
test9 = test9(:, sub2ind([1 size(test9, 2)], [indexCell2{:}], [indexCell{:}]));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N = 9;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Create training sets %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=0:N
    eval(['subtrain' num2str(i) '=train' num2str(i) '(1:round(size(train' num2str(i) ',1)*5/6),:);']);
end

a = '';
for i=0:N-1
   a = strcat(a, ' size(subtrain',  num2str(i), ', 1) + ');
end
a = strcat(a,  ' size(subtrain9, 1);');
train_size = eval([a]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Append training sets %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
train_data = [];

for i=0:N
    eval(['train_data=[train_data; subtrain' num2str(i) '];']);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Create training labels %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=0:N
    eval(['label' num2str(i) '=repmat(' num2str(i) ', size(subtrain' num2str(i) ', 1), 1);']);
end


a = '[';
for i=0:N-1
   a = strcat(a, 'label',  num2str(i), ';');
end
a = strcat(a, 'label',  num2str(9), '];');
train_label = eval([a]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Convert to double and normalize training data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
train_data = double(train_data);
% train_data = normc(train_data);
train_data = train_data/255;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Create validation sets %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=0:N
    eval(['validation' num2str(i) '=train' num2str(i) '(round(size(train' num2str(i) ',1)*5/6) + 1:end,:);']);
end

a = '';
for i=0:N-1
   a = strcat(a, ' size(validation',  num2str(i), ', 1) + ');
end
a = strcat(a,  ' size(validation9, 1);');
validation_size = eval([a]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Append validation sets %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
validation_data = [];

for i=0:N
    eval(['validation_data=[validation_data; validation' num2str(i) '];']);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Create validation labels %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=0:N
    eval(['label' num2str(i) '=repmat(' num2str(i) ', size(validation' num2str(i) ', 1), 1);']);
end


a = '[';
for i=0:N-1
   a = strcat(a, 'label',  num2str(i), ';');
end
a = strcat(a, 'label',  num2str(9), '];');
validation_label = eval([a]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Convert to double and normalize validation data %%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
validation_data = double(validation_data);
% validation_data = normc(validation_data);
validation_data = validation_data/255;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Create test sets %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=0:N
    eval(['subtest' num2str(i) '=test' num2str(i) '(1:round(size(test' num2str(i) ',1)*1/1),:);']);
end

a = '';
for i=0:N-1
   a = strcat(a, ' size(subtest',  num2str(i), ', 1) + ');
end
a = strcat(a,  ' size(subtest9, 1);');
test_size = eval([a]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Append test sets %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test_data = [];

for i=0:N
    eval(['test_data=[test_data; subtest' num2str(i) '];']);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Create test labels %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=0:N
    eval(['label' num2str(i) '=repmat(' num2str(i) ', size(subtest' num2str(i) ', 1), 1);']);
end


a = '[';
for i=0:N-1
   a = strcat(a, 'label',  num2str(i), ';');
end
a = strcat(a, 'label',  num2str(9), '];');
test_label = eval([a]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Convert to double and normalize test data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test_data = double(test_data);
% test_data = normc(test_data);
test_data = test_data/255;
end

