% discrete_continuous_info(d, c) estimates the mutual information between a
% discrete vector 'd' and a continuous vector 'c' using
% nearest-neighbor statistics.  Similar to the estimator described by
% Kraskov et. al. ("Estimating Mutual Information", PRE 2004)
% Each vector in c & d is stored as a column:  
% size(c) = (vector length, # samples).

function [f, V] = discrete_continuous_info(d, c, k, base)

if ~exist('k', 'var'), k = 3; end
if ~exist('base', 'var'), base = exp(1); end

first_symbol = [];
symbol_IDs = zeros(1, size(d, 2));
c_split = {};
cs_indices = {};
num_d_symbols = 0;


    % First, bin the continuous data 'c'
    % according to the discrete symbols 'd'

for c1 = 1:size(d, 2)
    symbol_IDs(c1) = num_d_symbols+1;
    for c2 = 1:num_d_symbols
        if d(:, c1) == d(:, first_symbol(c2))
            symbol_IDs(c1) = c2;
            break;
        end
    end
    if symbol_IDs(c1) > num_d_symbols
        num_d_symbols = num_d_symbols+1;
        first_symbol(num_d_symbols) = c1;
        c_split{num_d_symbols} = [];
        cs_indices{num_d_symbols} = [];
    end
    c_split{symbol_IDs(c1)} = [ c_split{symbol_IDs(c1)} c(:, c1) ];
    cs_indices{symbol_IDs(c1)} = [ cs_indices{symbol_IDs(c1)} c1 ];
end



    % Second, compute the neighbor statistic for each data pair (c, d)
    % using the binned c_split list
    
m_tot = 0;
av_psi_Nd = 0;
V = zeros(1, size(d, 2));
all_c_distances = zeros(1, size(c, 2));
psi_ks = 0;

for c_bin = 1:num_d_symbols
    
    one_k = min(k, size(c_split{c_bin}, 2)-1);
    
    if one_k > 0
        
        c_distances = zeros(1, size(c_split{c_bin}, 2));
        for pivot = 1:size(c_split{c_bin}, 2)
            
                % find the radius of our volume using only those samples with
                % the particular value of the discrete symbol 'd'
            
            for cv = 1:size(c_split{c_bin}, 2)
                vector_diff = c_split{c_bin}(:, cv) - c_split{c_bin}(:, pivot);
                c_distances(cv) = sqrt(vector_diff'*vector_diff);
            end
            sorted_distances = sort(c_distances);
            eps_over_2 = sorted_distances(one_k+1);   % don't count pivot
            
                % count the number of total samples within our volume using all
                % samples (all values of 'd')
            
            for cv = 1:size(c, 2)
                vector_diff = c(:, cv) - c_split{c_bin}(:, pivot);
                all_c_distances(cv) = sqrt(vector_diff'*vector_diff);
            end
            m = max(sum(all_c_distances <= eps_over_2) - 1, 0);   % don't count pivot
            
            m_tot = m_tot + psi(m);
            V(cs_indices{c_bin}(pivot)) = (2*eps_over_2)^size(d, 1);
        end
        
    else
        m_tot = m_tot + psi(num_d_symbols*2);
    end
    
    p_d = size(c_split{c_bin}, 2)/size(d, 2);
    av_psi_Nd = av_psi_Nd + p_d*psi(p_d*size(d, 2));
    psi_ks = psi_ks + p_d * psi(max(one_k, 1));
    
end

f = (psi(size(d, 2)) - av_psi_Nd + psi_ks - m_tot/size(d, 2)) / log(base);

end
