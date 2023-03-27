clc;
load('classes.mat');
load('cmap.mat');
clsIdx = 1:numel(classes);
clsMap = containers.Map(classes, clsIdx);

% read in the image and the scribble annotation
im = imread('2008_003147.jpg');
spots = readspots('2008_003147.xml', clsMap);

% begin to visualize the scribbles
numScribble = numel(unique(spots(:, 4)));
clsScribble = unique(spots(:, 3));
clsScribble = [clsScribble ones(numel(clsScribble), 1)];
him = image(im);
set(him, 'AlphaData', 0.5);
hold on;
axis off;
for ii = 1:numScribble
    scribble = spots(spots(:, 4) == ii, :);
    plot(scribble(:, 1), scribble(:, 2), 'Color', cmap(scribble(1, 3), :), ...
        'LineWidth', 3);
    idx = find(clsScribble(:, 1) == scribble(1, 3));
    if clsScribble(idx, 2) == 1
%         meanX = round(mean(scribble(:, 1)));
%         meanY = round(mean(scribble(:, 2)));
        text(scribble(1, 1)+5, scribble(1, 2)+5, classes{scribble(1, 3)}, 'Color', ...
            [0 0 0], 'FontSize', 13);
        clsScribble(idx, 2) = 0;
    end
end
