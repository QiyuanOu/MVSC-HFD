% discrete_continuous_info(d, c) estimates the mutual information between a
% discrete variable 'd' and a continuous variable 'c' using
% nearest-neighbor statistics.  Similar to the estimator described by
% Kraskov et. al. ("Estimating Mutual Information", PRE 2004)

function [f, V] = discrete_continuous_info_fast(d, c, k, base)

if ~exist('k', 'var'), k = 3; end
if ~exist('base', 'var'), base = exp(1); end

first_symbol = [];
symbol_IDs = zeros(1, length(d));
c_split = {};
cs_indices = {};
num_d_symbols = 0;


    % Sort the lists by the continuous variable 'c'

[c, c_idx] = sort(c);
d = d(c_idx);


    % Bin the continuous data 'c' according to the discrete symbols 'd'

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
        c_split{num_d_symbols} = [];
        cs_indices{num_d_symbols} = [];
    end
    c_split{symbol_IDs(c1)} = [ c_split{symbol_IDs(c1)} c(c1) ];
    cs_indices{symbol_IDs(c1)} = [ cs_indices{symbol_IDs(c1)} c1 ];
end



    % Compute the neighbor statistic for each data pair (c, d)
    % using the binned c_split list
    
m_tot = 0;
av_psi_Nd = 0;
V = zeros(1, length(d));
psi_ks = 0;

for c_bin = 1:num_d_symbols
    
    one_k = min(k, size(c_split{c_bin}, 2)-1);
    
    if one_k > 0
        
        for pivot = 1:length(c_split{c_bin})

                % find the radius of our volume using only those samples with
                % the particular value of the discrete symbol 'd'

            left_neighbor = pivot;
            right_neighbor = pivot;
            one_c = c_split{c_bin}(pivot);
            for ck = 1:one_k
                if left_neighbor == 1
                    right_neighbor = right_neighbor+1;
                    the_neighbor = right_neighbor;
                elseif right_neighbor == length(c_split{c_bin})
                    left_neighbor = left_neighbor-1;
                    the_neighbor = left_neighbor;
                elseif abs(c_split{c_bin}(left_neighbor-1) - one_c) < ...
                       abs(c_split{c_bin}(right_neighbor+1) - one_c)
                    left_neighbor = left_neighbor-1;
                    the_neighbor = left_neighbor;
                else
                    right_neighbor = right_neighbor+1;
                    the_neighbor = right_neighbor;
                end
            end
            distance_to_neighbor = abs(c_split{c_bin}(the_neighbor) - one_c);

                % count the number of total samples within our volume using all
                % samples (all values of 'd')

            if the_neighbor == left_neighbor
                m = floor( findpt(c, one_c + distance_to_neighbor) - ...
                    findpt(c, c_split{c_bin}(left_neighbor)) );
            else
                m = floor( findpt(c, c_split{c_bin}(right_neighbor)) - ...
                    findpt(c, one_c - distance_to_neighbor) );
            end
            if m < one_k
                m = one_k;
            end

            m_tot = m_tot + psi(m);
            V(cs_indices{c_bin}(pivot)) = 2 * distance_to_neighbor;
        end
        
    else
        m_tot = m_tot + psi(num_d_symbols*2);
        V(cs_indices{c_bin}(1)) = 2 * (c(end) - c(1));
    end
    
    p_d = length(c_split{c_bin})/length(d);
    av_psi_Nd = av_psi_Nd + p_d*psi(p_d*length(d));
    psi_ks = psi_ks + p_d * psi(max(one_k, 1));
    
end

% global dbgHx dbgHy dbgHxy
% for c_bin = 1:num_d_symbols
%     dps(c_bin) = length(c_split{c_bin})/length(d);
% end
% dbgHx = dbgHx - sum(dps.*log(dps));
% dbgHy = dbgHy + psi(length(d)) - m_tot/length(d) + mean(log(V));
% dbgHxy = dbgHxy + av_psi_Nd - psi_ks + mean(log(V));

f = (psi(length(d)) - av_psi_Nd + psi_ks - m_tot/length(d)) / log(base);

end


% findpt() finds the data point whose the value for the continuous
% variable is closest to 'c'.

function pt = findpt(c, target)

left = 1;
right = length(c);

if target < c(left)
    pt = 0.5;
    return;
elseif target > c(right)
    pt = right+0.5;
    return;
end

while left ~= right
    pt = floor((left+right)/2);
    if c(pt) < target
        left = pt;
    else
        right = pt;
    end
    
    if left+1 == right
        if c(left) == target
            pt = left;
        elseif c(right) == target
            pt = right;
        else
            pt = (right+left)/2;
        end
        break;
    end
end

end
