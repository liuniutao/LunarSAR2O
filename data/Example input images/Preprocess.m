clear; clc;

% Public demo script for converting one raw SAR group from .mat to .png.
% Change this value if you want to preprocess another example folder.
groupName = 'v1';

scriptDir = fileparts(mfilename('fullpath'));
rootDir = fullfile(scriptDir, groupName);

if ~exist(rootDir, 'dir')
    error('Input folder does not exist: %s', rootDir);
end

for k = 1:9
    name = sprintf('p%d', k);
    matPath = fullfile(rootDir, [name, '.mat']);
    outPath = fullfile(rootDir, [name, '.png']);

    if ~exist(matPath, 'file')
        fprintf('Missing file: %s\n', matPath);
        continue;
    end

    data = load(matPath);

    if ~isfield(data, name)
        fprintf('Variable %s not found in %s, skipping.\n', name, matPath);
        continue;
    end

    imgData = data.(name);

    if isempty(imgData) || ~isnumeric(imgData) || ndims(imgData) ~= 2
        fprintf('Invalid 2D numeric matrix in %s, skipping.\n', matPath);
        continue;
    end

    img = 10 * log10(max(imgData, eps));
    img = mat2gray(img, [-20 0]);
    imwrite(img, outPath);

    fprintf('Saved: %s\n', outPath);
end

disp('Preprocess finished.');
