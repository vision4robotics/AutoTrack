function [range_h,range_w,reg_window]=init_regwindow(sz,target_sz,params)
        reg_scale =target_sz;
        use_sz = sz;    
        reg_window = ones(use_sz) * params.reg_window_max;
        range = zeros(numel(reg_scale), 2);
    
        % determine the target center and range in the regularization windows
        for j = 1:numel(reg_scale)
            range(j,:) = [0, reg_scale(j) - 1] - floor(reg_scale(j) / 2);
        end
        center = floor((use_sz + 1)/ 2) + mod(use_sz + 1,2);
        range_h = (center(1)+ range(1,1)) : (center(1) + range(1,2));
        range_w = (center(2)+ range(2,1)) : (center(2) + range(2,2));   
        reg_window(range_h, range_w) = params.reg_window_min;


