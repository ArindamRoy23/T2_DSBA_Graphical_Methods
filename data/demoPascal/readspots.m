function spots = readspots(xml_file, cls_map)
    anno = xml2struct(xml_file);
    h = str2double(anno.annotation.size.height.Text);
    w = str2double(anno.annotation.size.width.Text);
    anno = anno.annotation.polygon;
    if ~iscell(anno)
        temp = anno;
        anno = cell(1, 1);
        anno{1} = temp;
    end
    spots = [];
    for jj = 1:numel(anno)
        cls = cls_map(anno{jj}.tag.Text);
        points = anno{jj}.point; 
        if ~iscell(points)
            temp = points;
            points = cell(1, 1);
            points{1} = temp;
        end
        for kk = 1:numel(points)
            x = str2double(points{kk}.X.Text)+1;
            y = str2double(points{kk}.Y.Text)+1;
            x = max(min(x, w), 1);
            y = max(min(y, h), 1);
            spots = [spots; x y cls jj];
        end
    end
end