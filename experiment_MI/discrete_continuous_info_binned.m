% discrete_continuous_info_binned(d, c) estimates the mutual information
% between a discrete variable 'd' and a continuous variable 'c' by
% binning in variable 'c'.  There are a fixed number of points in each bin.

function MI = discrete_continuous_info_binned(d, c, bin_size, base)

if ~exist('bin_size', 'var'), bin_size = 5; end
if ~exist('base', 'var'), base = exp(1); end

first_symbol = [];
symbol_IDs = zeros(1, length(d));
num_d_symbols = 0;


    % First, identify each discrete sample 'd'

for c1 = 1:length(d)
    symbol_IDs(c1) = num_d_symbols+1;
    for c2 = 1:num_d_symbols
        if d(c1) == d(first_symbol(c2))
            symbol_IDs(c1) = c2;
            break;
        end
    end
    if symbol_IDs(c1) > num_d_symbols
        num_d_symbols = num_d_symbols+1;
        first_symbol(num_d_symbols) = c1;
    end
end


    % Construct bins for the continuous variable 'c'
    
num_c_bins = ceil(length(c) / bin_size);

[c_sorted, sort_idx] = sort(c);
d_sorted = symbol_IDs(sort_idx);

first_bin_idx = bin_size*(1:num_c_bins-1);
bin_boundaries = (c_sorted(first_bin_idx) + c_sorted(first_bin_idx+1)) / 2;
bin_boundaries = [ c_sorted(1) - (c_sorted(2)-c_sorted(1))/2, ...
    bin_boundaries, c_sorted(end) + (c_sorted(end)-c_sorted(end-1))/2 ];
bin_widths = bin_boundaries(2:end) - bin_boundaries(1:end-1);


    % fill in the binned frequency table 'freq' and calculate the MI
    
freq = zeros(num_d_symbols, num_c_bins);

for cs = 1:length(c)
    c_idx = floor((cs-1)/bin_size)+1;
    freq(d_sorted(cs), c_idx) = freq(d_sorted(cs), c_idx) + 1;
end
freq = freq / length(c);

px = sum(freq, 2);
py = sum(freq, 1);
pxy = freq;
pxy = pxy + (pxy == 0);    % a hack:  makes the limit "p log p = 0 at p = 0" work

MI = sum(sum(freq.*log(pxy))) - sum(px.*log(px)) - sum(sum(freq, 1).*log(py));
MI = MI / log(base);

end
