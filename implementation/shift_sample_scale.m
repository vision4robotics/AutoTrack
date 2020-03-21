function out = shift_sample_scale(im, pos, base_target_sz,xs,recovered_scale,scaleFactors,scale_window,scale_model_sz)
        nScales = length(scaleFactors);
        out = zeros(size(xs,1), nScales, 'single');
        shift_pos=recovered_scale-ceil(nScales/2);
        if shift_pos==0
            out=xs;
        elseif  shift_pos>0
            for j=1:nScales-shift_pos
            out(:,j)=xs(:,j+shift_pos)/(scale_window(j+shift_pos)+1e-5)*scale_window(j);
            end
            for i=1:shift_pos
                 patch_sz = floor(base_target_sz * scaleFactors(nScales-shift_pos+i));
				 patch_sz = max(patch_sz, 2);
                 xs = floor(pos(2)) + (1:patch_sz(2)) - floor(patch_sz(2)/2);
                 ys = floor(pos(1)) + (1:patch_sz(1)) - floor(patch_sz(1)/2);
                 % check for out-of-bounds coordinates, and set them to the values at
                 % the borders
                xs(xs < 1) = 1;
                ys(ys < 1) = 1;
                xs(xs > size(im,2)) = size(im,2);
                ys(ys > size(im,1)) = size(im,1);
    
                % extract image
                im_patch = im(ys, xs, :);
    
                % resize image to model size
                im_patch_resized = mexResize(im_patch, scale_model_sz, 'auto');
    
                % extract scale features
                temp_hog = fhog(single(im_patch_resized), 4);
                temp = temp_hog(:,:,1:31);
                % window
                out(:,nScales-shift_pos+i) = temp(:) * scale_window(nScales-shift_pos+i);
            end  
        else
            for j=1:nScales+shift_pos
            out(:,j-shift_pos)=xs(:,j)/(scale_window(j)+1e-5).*scale_window(j-shift_pos);
            end
            for i=1:-shift_pos
                 patch_sz = floor(base_target_sz * scaleFactors(i));
				 patch_sz = max(patch_sz, 2);
                 xs = floor(pos(2)) + (1:patch_sz(2)) - floor(patch_sz(2)/2);
                 ys = floor(pos(1)) + (1:patch_sz(1)) - floor(patch_sz(1)/2);
                 % check for out-of-bounds coordinates, and set them to the values at
                 % the borders
                xs(xs < 1) = 1;
                ys(ys < 1) = 1;
                xs(xs > size(im,2)) = size(im,2);
                ys(ys > size(im,1)) = size(im,1);
    
                % extract image
                im_patch = im(ys, xs, :);
    
                % resize image to model size
                im_patch_resized = mexResize(im_patch, scale_model_sz, 'auto');
    
                % extract scale features
                temp_hog = fhog(single(im_patch_resized), 4);
                temp = temp_hog(:,:,1:31);
                % window
                out(:,i) = temp(:) * scale_window(i);
            end
        end

