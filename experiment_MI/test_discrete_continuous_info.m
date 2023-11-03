function test_discrete_continuous_info()

rng(0)      % uncomment for reproducible results


% --------------- Adjustable parameters ---------------

compare_slow_calculator = false;


    % square wave test params

a = [ 0 .1 .2 ];            % the left side of each square wave
b_a = [ 1 1.1 1.1 ];        % the length in y of each square wave
p{1} = [ .2 1 0.5 ];        % the (normalized) amplitude p(x)

    % gaussian test params

y0 = [ .4 .5 .8 ];          % the center of the gaussian
sigma_y = [ .2 .3 .25 ];    % the gaussian decay constant
p{2} = [ .2 1 0.5 ];        % the (normalized) amplitude p(x)


k = 1:10;                   % the kth neighbor
bin_size = 0:.02:1;         % # data points per bin
experiment_type = [ 1 2 1 2 ];           % 1 = square wave; 2 = gaussian
sample_size = [ 10000 10000 400 400 ];   % number of data points
reps = 100;                 % number of times to repeat each sample

dist_bins = 30;             % number of bins used to plot data points
fig_number = [4 2];         % output figures




% --------------- The test... ---------------


for ce = 1:length(p)
    num_x(ce) = length(p{ce});
    p{ce} = p{ce} / sum(p{ce});
    cumulative_p{ce} = cumsum(p{ce});
end
    
for ce = 1:length(experiment_type)
    bin_k{ce} = floor(sample_size(ce).^bin_size);
    [ bin_k{ce}, unique_indices, dummy ] = unique(bin_k{ce}, 'stable');
    unique_bin_size{ce} = bin_size(unique_indices);
end

I_compare = 0;
V_compare = 0;




% --------------- Calculate the true mutual information ---------------

true_info = [ 0 0 ];


domains = sort([ a a+b_a ]);       % MI of square wave PDFs

for cd = 1:length(domains)-1
if domains(cd) < domains(cd+1)
    p_domain = 0;
    for cx = 1:num_x(1)
        if a(cx) <= domains(cd) && a(cx)+b_a(cx) >= domains(cd+1)
            p_domain = p_domain + p{1}(cx) / b_a(cx);
        end
    end
    if p_domain > 0
        true_info(1) = true_info(1) - (domains(cd+1) - domains(cd)) * ...
            p_domain * log(p_domain);
    end
end, end

for cx = 1:num_x(1)
    true_info(1) = true_info(1) - p{1}(cx) * log(b_a(cx));
end


Ay = p{2}./(sqrt(2*pi)*sigma_y);     % MI of Gaussians
mu_y = @(y) sum( Ay .* exp(-(y-y0).^2./(2*sigma_y.^2)), 2 );
Hy = @(y) -mu_y(y) .* log(mu_y(y));

true_info(2) = integral(Hy, min(y0 - 10*sigma_y), max(y0 + 10*sigma_y), ...
        'ArrayValued', true) - 0.5 - sum(p{2}.*log(sqrt(2*pi)*sigma_y));

true_info = true_info / log(2);



% --------------- Sample distributions and estimate MI ---------------


for experiment = 1:length(sample_size)
    
    disp(['Experiment ', num2str(experiment)])
    
    nn_info{experiment} = zeros(reps, length(k));
    nn_info_err{experiment} = nn_info{1};
    averaging_err{experiment} = nn_info{1};
    binned_info{experiment} = zeros(reps, length(bin_k{experiment}));
    binned_info_err{experiment} = binned_info{experiment};
    
    samples = sample_size(experiment);
    V = zeros(1, samples);
    expt_type = experiment_type(experiment);
    
    for rep = 1:reps
        
        
            % generate the samples:  (x_vec, y_vec)
        
        x_vec = zeros(1, samples);
        y_vec = x_vec;
        xy = zeros(num_x(expt_type), samples);
        ns = zeros(1, num_x(expt_type));
        
        for n = 1:samples
            idxs = find(rand() < cumulative_p{expt_type});
            x = idxs(1);
            x_vec(n) = x;
            
            if experiment_type(experiment) == 1
                y_vec(n) = rand()*b_a(x) + a(x);
            elseif experiment_type(experiment) == 2
                y_vec(n) = sigma_y(x)*randn() + y0(x);
            end
            ns(x) = ns(x)+1;
            xy(x, ns(x)) = y_vec(n);
        end
        
        
        for ck = 1:length(k)
            
                % estimate MI using nearest neighbors
            
            [one_dc_info, V] = ...
                discrete_continuous_info_fast(x_vec, y_vec, k(ck), 2);
            nn_info{experiment}(rep, ck) = one_dc_info;
            nn_info_err{experiment}(rep, ck) = ...
                    abs(one_dc_info - true_info(expt_type));
            
            if compare_slow_calculator == true
                x_vec_2 = [ floor(x_vec/2); mod(x_vec, 2) ];
                y_vec_2 = [ y_vec; y_vec ];
                y_vec_2(ceil(rand()*2), :) = 0;
                [I_compare, V_compare] = ...
                    discrete_continuous_info(x_vec_2, y_vec_2, ck, 2);
                if abs(nn_info_err{experiment}(rep, ck) - I_compare) > .00001
                    disp( ['mismatch for ', num2str(samples), ...
                        ' data points, point #', num2str(rep) ]);
                end
            end
            
            
                % estimate the y-averaging error in case we are interested
            
            y_avg_error = 0*y_vec;
            for n = 1:samples
            if V(n) ~= 0
                py = 0;
                av_py = 0;
                for cx = 1:num_x(expt_type)
                    one_py = 1./b_a(cx);
                    if y_vec(n) < a(cx) || y_vec(n) > a(cx)+b_a(cx)
                        one_py = 0;
                    end
                    one_av_py = one_py * (min(y_vec(n)+V(n), a(cx)+b_a(cx)) ...
                                    - max(y_vec(n)-V(n), a(cx))) / (2*V(n));
                    if cx == x_vec(n)
                        py_x = one_py;
                        av_py_x = one_av_py;
                    end
                    py = py + p{expt_type}(cx)*one_py;
                    av_py = av_py + p{expt_type}(cx)*one_av_py;
                end
                y_avg_error(n) = log(py*av_py_x) - log(av_py*py_x);
            end, end
            
            averaging_err{experiment}(rep, ck) = mean(y_avg_error).^2;
        end
        
        
            % estimate MI using the binning method
            
        for ck = 1:length(bin_k{experiment})
            one_binned_info = discrete_continuous_info_binned(...
                x_vec, y_vec, bin_k{experiment}(ck), 2);
            binned_info{experiment}(rep, ck) = one_binned_info;
            binned_info_err{experiment}(rep, ck) = ...
                    abs(one_binned_info - true_info(expt_type));
        end
    end
    
    last_x_vec{experiment} = x_vec;
    last_y_vec{experiment} = y_vec;
end


% --------------- Generate output plots ---------------


distnames = { 'Square wave', 'Gaussian' };
allcolors = 'ygrbmck';

figure(fig_number(1)), clf
for ce = 1:length(experiment_type)
    
    
        % Plot 1A:  the distributions we sampled from
    
    subplot(length(experiment_type), 3, ce*3-2), hold on
    expt_type = experiment_type(ce);
    
    if experiment_type(ce) == 1
        
        plot_start = min(a) - 0.05*(max(a+b_a)-min(a));
        plot_end = max(a+b_a) + 0.05*(max(a+b_a)-min(a));
        for cx = 1:num_x(1)
            plot([plot_start, a(cx), a(cx), a(cx)+b_a(cx), a(cx)+b_a(cx), plot_end], ...
                [0, 0, p{1}(cx)/b_a(cx), p{1}(cx)/b_a(cx), 0, 0], allcolors(mod(cx, 7)+1), 'LineWidth', 2)
        end
        
        hist_centers = (-0.1:.1:1.3)+0.05;
        
    elseif experiment_type(ce) == 2
        
        y = min(y0 - 5*sigma_y) : 0.01*min(sigma_y) : max(y0 + 5*sigma_y);
        for cx = 1:num_x(2)
            plot(y, p{2}(cx)*exp(-(y-y0(cx)).^2 / (2*sigma_y(cx)^2)) ...
                / (sqrt(2*pi)*sigma_y(cx)), allcolors(mod(cx, 7)+1), 'LineWidth', 2);
        end
        
        hist_centers = (-0.6:.15:1.6)+0.05;
        
    end
    
    for cx = 1:num_x(expt_type)
        hist_heights = hist(last_y_vec{ce}(last_x_vec{ce} == cx), hist_centers);
%        plot(bin_centers, bin_heights/length(last_y_vec{ce}) ...
%                /(bin_centers(2)-bin_centers(1)), [allcolors(mod(cx, 7)+1) 'o'])
        stairs(hist_centers-(hist_centers(2)-hist_centers(1))/2, hist_heights/length(last_y_vec{ce}) ...
                /(hist_centers(2)-hist_centers(1)), allcolors(mod(cx, 7)+1))
    end
    
    title([distnames{experiment_type(ce)} '; ', num2str(sample_size(ce)), ' points / data set'])
    xlabel('y')
    ylabel('p(y|x)')
    axis([-0.7 1.7 0 1.0])
    
    
    
        % Plot 1B:  MI as a function of k using nearest neighbors
    
    subplot(length(experiment_type), 3, ce*3-1), hold on
    plot_with_error(k, nn_info{ce}, [0, 0, 0]);
    plot(k, 0*k+true_info(experiment_type(ce)), '--')
    title('Nearest neighbor estimate')
    xlabel('k')
    ylabel('MI (bits)')
    axis tight, ylim([0 0.4])
    
    
    
        % Plot 1C:  MI as a function of k using binning method
    
    subplot(length(experiment_type), 3, ce*3), hold on
    plot_with_error(unique_bin_size{ce}, binned_info{ce}, [0, 0, 0]);
    plot(unique_bin_size{ce}, 0*unique_bin_size{ce}+true_info(experiment_type(ce)), '--')
    title('Binning estimate')
    xlabel('log_{N} n')
    ylabel('MI (bits)')
    axis tight, ylim([0 0.4])
end

set(gcf, 'PaperPosition', [0.25, 2.5, [6.83, 6]])



    % Plot 2:  Difference in MI error between the two methods

plottitles = { '10,000 points / data set', '400 points / data set' };
xaxislabels = { 'k',  };
expt_colors = { 'r', 'b', [ .2 .2 .2 ], 'm' };
expt_comparisons = { { 1, 2 }, { 3, 4 } };
legends = { { 'square wave', 'Gaussian' }, { 'square wave', 'Gaussian' } };
plot_colors = { [.4, 0, .4], [0, .5, .5] };
k_choice = 3;

fig2 = figure(fig_number(2)); clf

for ce = 1:length(expt_comparisons)
    subplot(length(expt_comparisons), 1, ce), hold on
    
    for ce2 = 1:length(expt_comparisons{ce})
        one_expt = expt_comparisons{ce}{ce2};
        median_nn_err = median(nn_info_err{one_expt}, 1);
        plot(unique_bin_size{one_expt}, median(binned_info_err{one_expt}, 1) ...
                    / median_nn_err(k_choice), 'Color', plot_colors{ce2});
    end
    
    plot(unique_bin_size{ce}, 0*unique_bin_size{ce}+1, '--');
    
    title(plottitles{ce})
    xlabel('data points per bin (n)')
    ylabel('\DeltaI_{bin} / median(\DeltaI_{nn})')
    axis tight, ylim([0.25 10])
    set(get(fig2, 'CurrentAxes'), 'YScale', 'log');
    legend(legends{ce})
end

set(gcf, 'PaperPosition', [0.25, 2.5, [3.27, 6]])



disp(['True mutual info (sq wave): ', num2str(true_info(1))])
disp(['True mutual info (Gaussian): ', num2str(true_info(2))])



    function plot_with_error(x, y, c)
    
    y_median = median(y, 1);
    y_sorted = sort(y, 1);
    y10 = y_sorted(ceil(0.1*size(y_sorted, 1)), :);
    y90 = y_sorted(ceil(0.9*size(y_sorted, 1)), :);
    
    if size(y, 1) > 1
        patch([x, fliplr(x)], [y10, fliplr(y90)], min(c + 0.7*[1 1 1], 1), ...
            'EdgeColor', 'none');
    end
    plot(x, y_median, 'Color', c)
    
    end

end
