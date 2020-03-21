function xf = shift_sample(xf, shift, kx, ky)

% Shift a sample in the Fourier domain. The shift should be normalized to
% the range [-pi, pi].

shift_exp_y =exp((1i * shift(1)) * ky);
shift_exp_x = exp((1i * shift(2)) * kx);
xf =bsxfun(@times, bsxfun(@times, xf, shift_exp_y), shift_exp_x);