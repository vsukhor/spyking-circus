import matplotlib.colors as mcolors
import numpy
import scipy
import pylab
import os
import statsmodels.api as sm
from circus.shared.files import load_data, get_results, get_results
import numpy
import pylab
# from circus.shared import algorithms as algo
from circus.shared.utils import *
from circus.shared.parser import CircusParser
from circus.shared.probes import get_nodes_and_edges


def view_fit(file_name, t_start=0, t_stop=1, n_elec=2, fit_on=True, square=True, templates=None, save=False):
    
    params = CircusParser(file_name)
    data_file = params.get_data_file()
    data_file.open()
    _ = data_file.analyze(int(params.rate))
    N_e = params.getint('data', 'N_e')
    N_t = params.getint('detection', 'N_t')
    N_total = params.nb_channels
    sampling_rate = int(params.rate)
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening = params.getboolean('whitening', 'spatial')
    spike_thresh = params.getfloat('detection', 'spike_thresh')
    file_out_suff = params.get('data', 'file_out_suff')
    template_shift = params.getint('detection', 'template_shift')
    nodes, edges = get_nodes_and_edges(params)
    chunk_size = (t_stop - t_start) * sampling_rate
    padding = (t_start * sampling_rate, t_start * sampling_rate)
    suff = params.get('data', 'suffix')

    if do_spatial_whitening:
        spatial_whitening = load_data(params, 'spatial_whitening')
    if do_temporal_whitening:
        temporal_whitening = load_data(params, 'temporal_whitening')

    thresholds = load_data(params, 'thresholds')
    data, _ = data_file.get_data(0, chunk_size, padding=padding, nodes=nodes)
    data_shape = len(data)

    data_file.close()
    if do_spatial_whitening:
        data = numpy.dot(data, spatial_whitening)
    if do_temporal_whitening:
        data = scipy.ndimage.filters.convolve1d(data, temporal_whitening, axis=0, mode='constant')

    try:
        result = load_data(params, 'results')
    except Exception:
        result = {'spiketimes': {}, 'amplitudes': {}}
    if fit_on:
        curve = numpy.zeros((N_e, (t_stop - t_start) * sampling_rate), dtype=numpy.float32)
        count = 0
        limit = (t_stop - t_start) * sampling_rate - template_shift + 1
        if templates is None:
            try:
                templates = load_data(params, 'templates')
            except Exception:
                templates = numpy.zeros((0, 0, 0))
        for key in list(result['spiketimes'].keys()):
            elec = int(key.split('_')[1])
            lims = (t_start*sampling_rate + template_shift, t_stop*sampling_rate - template_shift-1)
            idx = numpy.where((result['spiketimes'][key] > lims[0]) & (result['spiketimes'][key] < lims[1]))
            for spike, (amp1, amp2) in zip(result['spiketimes'][key][idx], result['amplitudes'][key][idx]):
                count += 1
                spike -= t_start*sampling_rate
                tmp1 = templates[:, elec].toarray().reshape(N_e, N_t)
                tmp2 = templates[:, elec+templates.shape[1] // 2].toarray().reshape(N_e, N_t)

                curve[:, spike - template_shift:spike + template_shift + 1] += amp1 * tmp1 + amp2 * tmp2
        print(("Number of spikes %d" % count))

    if not numpy.iterable(n_elec):
        if square:
            idx = numpy.random.permutation(numpy.arange(N_e))[:n_elec**2]
        else:
            idx = numpy.random.permutation(numpy.arange(N_e))[:n_elec]
    else:
        idx = n_elec
        n_elec = len(idx)
    pylab.figure()
    for count, i in enumerate(idx):
        if square:
            pylab.subplot(n_elec, n_elec, count + 1)
            if numpy.mod(count, n_elec) != 0:
                pylab.setp(pylab.gca(), yticks=[])
            else:
                pylab.ylabel('Signal')
            if count < n_elec*(n_elec - 1):
                pylab.setp(pylab.gca(), xticks=[])
            else:
                pylab.xlabel('Time [ms]')
        else:
            pylab.subplot(n_elec, 1, count + 1)
            if count != (n_elec - 1):
                pylab.setp(pylab.gca(), xticks=[])
            else:
                pylab.xlabel('Time [ms]')

        pylab.plot(data[:, i], '0.25')
        if fit_on:
            pylab.plot(curve[i], 'r')
        xmin, xmax = pylab.xlim()
        pylab.plot([xmin, xmax], [-thresholds[i], -thresholds[i]], 'k--')
        pylab.plot([xmin, xmax], [thresholds[i], thresholds[i]], 'k--')
        pylab.title('Electrode %d' % i)
        if (square and not (count < n_elec*(n_elec - 1))) or (not square and not count != (n_elec - 1)):
            x, y = pylab.xticks()
            pylab.xticks(x, numpy.round(x // sampling_rate, 2))

        pylab.ylim(-2 * thresholds[i], 2 * thresholds[i])
    pylab.tight_layout()
    if save:
        pylab.savefig(os.path.join(save[0], save[1]))
        pylab.close()
    else:
        pylab.show()



def variance_template(template, channel_mads, mads, save=False):

    template = template.flatten()
    flat_channel_mads = channel_mads.flatten()
    frac_high_variances = channel_mads/mads[:, numpy.newaxis]

    fig = pylab.figure()

    # Centroids plot.
    ax = fig.add_subplot(211)
    ax.fill_between(numpy.arange(template.size), template-flat_channel_mads, template+flat_channel_mads, color='k', alpha=0.5)
    ax.plot(template, 'r', lw=2)
    ax.set_ylabel('Amplitude')
    ax.set_xticks([])

    ax = fig.add_subplot(212)
    ax.plot(frac_high_variances.flatten())
    ax.set_ylabel('Ratio variance')
    ax.set_title('Max ratio %g' %frac_high_variances.max())
    ax.set_xlabel('Time Steps')
    ax.plot([0, frac_high_variances.size], [1, 1], 'k--')

    if save:
        pylab.savefig(os.path.join(save[0], 'variance_' + save[1]))
        pylab.close()
    else:
        pylab.show()



def view_clusters(data, rho, delta, centers, halo, injected=None, save=False, alpha=3):
    """Sanity plot of the clusters (per electrode).

    Arguments:
        data
        rho
        delta
        centers
            Indices of the points which correspond to the centroids.
        halo
        injected (optional)
        save: (optional)
        alpha: (optional)
    """

    marker_size = 3 ** 2
    my_cmap = pylab.get_cmap('jet')
    c_norm = mcolors.Normalize(vmin=numpy.min(halo), vmax=numpy.max(halo))
    scalar_map = pylab.cm.ScalarMappable(norm=c_norm, cmap=my_cmap)
    assigned = numpy.where(halo > -1)[0]
    not_assigned = numpy.where(halo == -1)[0]
    colors = scalar_map.to_rgba(halo[assigned])

    fig = pylab.figure(figsize=(15, 10))

    # Centroids plot.
    ax = fig.add_subplot(241)
    # # Plot points.
    ax.scatter(rho, delta, s=marker_size, c='k', linewidths=0)
    # # Highlight the centroids.
    for i in centers:
        if halo[i] > -1:
            color_val = scalar_map.to_rgba(halo[i])
            ax.scatter([rho[i]], [delta[i]], s=marker_size, c=color_val, linewidths=0)
    # # Adjust axis.
    rho_min, rho_max = ax.get_xlim()
    try:
        ax.set_yscale('log')
    except Exception:
        pass
    # # Add labels.
    ax.set_xlim(rho_min, rho_max)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$\delta$")
    ax.set_title("Centroids")

    # Plot PC 1 vs PC 0.
    ax = fig.add_subplot(242)
    ax.set_aspect('equal')
    ax.scatter(data[not_assigned, 0], data[not_assigned, 1], s=marker_size, c='k', alpha=0.5, linewidths=0)
    ax.scatter(data[assigned, 0], data[assigned, 1], s=marker_size, c=colors, cmap=my_cmap, linewidths=0)
    ax.axhline(y=0.0, color='k', alpha=0.5, linewidth=0.3)
    ax.axvline(x=0.0, color='k', alpha=0.5, linewidth=0.3)
    ax.set_xlabel("dim. 0")
    ax.set_ylabel("dim. 1")

    if data.shape[1] > 2:

        # Plot PC 2 vs PC 0.
        ax = fig.add_subplot(243)
        ax.set_aspect('equal')
        ax.scatter(data[not_assigned, 0], data[not_assigned, 2], s=marker_size, c='k', alpha=0.5, linewidths=0)
        ax.scatter(data[assigned, 0], data[assigned, 2], s=marker_size, c=colors, cmap=my_cmap, linewidths=0)
        ax.axhline(y=0.0, color='k', alpha=0.5, linewidth=0.3)
        ax.axvline(x=0.0, color='k', alpha=0.5, linewidth=0.3)
        ax.set_xlabel("dim. 0")
        ax.set_ylabel("dim. 2")

        # Plot PC 2 vs PC 1.
        ax = fig.add_subplot(244)
        ax.set_aspect('equal')
        ax.scatter(data[not_assigned, 1], data[not_assigned, 2], s=marker_size, c='k', alpha=0.5, linewidths=0)
        ax.scatter(data[assigned, 1], data[assigned, 2], s=marker_size, c=colors, cmap=my_cmap, linewidths=0)
        ax.axhline(y=0.0, color='k', alpha=0.5, linewidth=0.3)
        ax.axvline(x=0.0, color='k', alpha=0.5, linewidth=0.3)
        ax.set_xlabel("dim. 1")
        ax.set_ylabel("dim. 2")

    my_cmap = pylab.get_cmap('winter')

    # Rho plot.
    ax = fig.add_subplot(247)
    ax.set_aspect('equal')
    idx = numpy.argsort(rho[assigned])
    ax.scatter(
        data[assigned[idx], 0], data[assigned[idx], 1], s=marker_size, c=rho[assigned[idx]], cmap=my_cmap, linewidths=0
    )
    ax.scatter(data[centers, 0], data[centers, 1], s=marker_size, c='r', linewidths=0)
    if injected is not None:
        ax.scatter(data[injected, 0], data[injected, 1], s=marker_size, c='b', linewidths=0)
    ax.axhline(y=0.0, color='k', alpha=0.5, linewidth=0.3)
    ax.axvline(x=0.0, color='k', alpha=0.5, linewidth=0.3)
    ax.set_xlabel("dim. 0")
    ax.set_ylabel("dim. 1")
    ax.set_title(r"$\rho$")

    # Delta plot.
    ax = fig.add_subplot(248)
    ax.set_aspect('equal')
    idx = numpy.argsort(delta[assigned])
    ax.scatter(
        data[assigned[idx], 0], data[assigned[idx], 1], s=marker_size,
        c=numpy.log(1 + delta[assigned[idx]]), cmap=my_cmap, linewidths=0
    )
    ax.scatter(data[centers, 0], data[centers, 1], s=marker_size, c='r', linewidths=0)
    if injected is not None:
        ax.scatter(data[injected, 0], data[injected, 1], c='b', linewidths=0)
    ax.axhline(y=0.0, color='k', alpha=0.5, linewidth=0.3)
    ax.axvline(x=0.0, color='k', alpha=0.5, linewidth=0.3)
    ax.set_xlabel("dim. 0")
    ax.set_ylabel("dim. 1")
    ax.set_title(r"$\delta$")

    # Putative centroids plot.
    ax = fig.add_subplot(245)
    # # Plot points.
    ax.scatter(rho, delta, s=marker_size, c='k', linewidths=0)

    # Preliminary computations.
    z_score = None
    try:
        x = sm.add_constant(rho)
        model = sm.RLM(delta, x)
        results = model.fit()
        difference = delta - results.fittedvalues
        factor = numpy.median(numpy.abs(difference - numpy.median(difference)))
        upper = results.fittedvalues + alpha * factor * (1 + results.fittedvalues)
        z_score = difference - alpha * factor * (1 + results.fittedvalues)
        # # Plot excluded region.
        idx = numpy.argsort(rho)
        ax.fill_between(rho[idx], results.fittedvalues[idx], y2=upper[idx], alpha=0.5, color='r')
    except Exception:
        pass

    # TODO remove the following commented block (deprecated)?
    # # # Highlight the centroids.
    # m_centers = numpy.where(z_score >= 0)[0]
    # ax.plot(rho[m_centers], delta[m_centers], 'r.')
    # # Highlight the centroids.
    for i in centers:
        if halo[i] > -1:
            color_val = scalar_map.to_rgba(halo[i])
            ax.scatter([rho[i]], [delta[i]], s=marker_size, c=color_val, linewidths=0)
    # # Adjust axis.
    try:
        ax.set_yscale('log')
    except Exception:
        pass
    ax.set_xlim(rho_min, rho_max)
    # # Add labels.
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$\delta$")
    ax.set_title("Putative centroids")

    # Putative centroids plot.
    ax = fig.add_subplot(246)
    if z_score is not None:
        ax.scatter(rho, z_score, s=marker_size, c='k', linewidths=0)
        # # Highlight the centroids.
        for i in centers:
            if halo[i] > -1:
                color_val = scalar_map.to_rgba(halo[i])
                ax.scatter([rho[i]], [z_score[i]], s=marker_size, c=color_val, linewidths=0)
        # # Adjust axis.
        ax.set_xlim(rho_min, rho_max)
        # # Add labels.
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(r"$\epsilon$")
        ax.set_title("Putative centroids")

    try:
        pylab.tight_layout()
    except Exception:
        pass

    if save:
        try:
            pylab.savefig(os.path.join(save[0], 'cluster_%s' % save[1]))
            pylab.close()
        except Exception:
            pass
    else:
        pylab.show()

    del fig

    return


def view_local_merges(
        waveforms_data, clusters_data, old_allocation, new_allocation, merge_history,
        save=False, max_nb_traces=200,
):

    import matplotlib.pyplot as plt

    marker_size = 10
    color_map = plt.get_cmap('jet')

    local_merges = merge_history['merge']
    nb_local_merges = len(local_merges)
    cluster_distances = merge_history['distance']
    merging_method = merge_history['method']
    merging_threshold = merge_history['threshold']
    allocation = np.copy(old_allocation)
    _ = new_allocation

    if nb_local_merges > 0:

        # Compute the number of groups of local merges.
        local_merge_groups = {}
        local_merge_flat_groups = {}
        local_merge_distances = {}
        for local_merge, cluster_distance in zip(local_merges, cluster_distances):
            cluster_nb_1, cluster_nb_2 = local_merge
            # 1.
            if cluster_nb_1 not in local_merge_groups:
                group_1 = cluster_nb_1
            else:
                group_1 = local_merge_groups[cluster_nb_1]
                del local_merge_groups[cluster_nb_1]
            if cluster_nb_2 not in local_merge_groups:
                group_2 = cluster_nb_2
            else:
                group_2 = local_merge_groups[cluster_nb_2]
                del local_merge_groups[cluster_nb_2]
            local_merge_groups[cluster_nb_1] = (group_1, group_2)
            # 2.
            local_merge_distances[cluster_nb_1] = None
            local_merge_distances[cluster_nb_2] = cluster_distance
            # 3.
            if cluster_nb_1 not in local_merge_flat_groups:
                group_1 = [cluster_nb_1]
            else:
                group_1 = local_merge_flat_groups[cluster_nb_1]
                del local_merge_flat_groups[cluster_nb_1]
            if cluster_nb_2 not in local_merge_flat_groups:
                group_2 = [cluster_nb_2]
            else:
                group_2 = local_merge_flat_groups[cluster_nb_2]
                del local_merge_flat_groups[cluster_nb_2]
            local_merge_flat_groups[cluster_nb_1] = group_1 + group_2
        assert len(local_merge_groups) == len(local_merge_flat_groups)
        nb_local_merge_groups = len(local_merge_groups)
        # Compute the maximal number of local merges in one of these groups.
        max_nb_clusters_per_group = 2
        for cluster_nb in list(local_merge_flat_groups.keys()):
            max_nb_clusters_per_group = max(max_nb_clusters_per_group, len(local_merge_flat_groups[cluster_nb]))

        nb_rows = nb_local_merge_groups
        nb_columns = 1 + 1 + max_nb_clusters_per_group
        fig_width, fig_height = plt.rcParams['figure.figsize']
        figsize = float(nb_columns) * (0.33 * fig_width), float(nb_rows) * (0.5 * fig_height)

        cluster_nbs = np.unique(old_allocation[old_allocation > - 1])
        colors = {
            cluster_nb: mcolors.to_rgb('C{}'.format(k % 10))
            for k, cluster_nb in enumerate(cluster_nbs)
        }

        fig, axes = plt.subplots(nrows=nb_rows, ncols=nb_columns, figsize=figsize)
        axes = axes.reshape(nb_rows, nb_columns)
        for ax in axes.flatten():
            ax.set_axis_off()

        # Prepare 2D projection plot.
        is_assigned = numpy.where(allocation > -1)[0]
        is_not_assigned = numpy.where(allocation == -1)[0]
        # Plot 2D projection.
        ax = axes[0, 0]
        ax.set_axis_on()
        x = clusters_data[is_not_assigned, 0]
        y = clusters_data[is_not_assigned, 1]
        ax.scatter(x, y, c='k', linewidth=0, s=marker_size, alpha=0.5)
        x = clusters_data[is_assigned, 0]
        y = clusters_data[is_assigned, 1]
        c = np.array([
            colors[cluster_nb]
            for cluster_nb in allocation[is_assigned]
        ])
        ax.scatter(x, y, c=c, cmap=color_map, linewidth=0, s=marker_size)
        nb_clusters = len(np.unique(allocation[is_assigned]))
        ax.annotate(
            "{:d} clusters".format(nb_clusters),
            xy=(0, 0), xycoords='axes fraction',
            xytext=(2, 1), textcoords='offset points',
            horizontalalignment='left', verticalalignment='bottom'
        )
        ax.set_aspect('equal')
        ax.set_xlabel('dim. 0')
        ax.set_ylabel('dim. 1')
        ax.set_title('2D projection')

        for row_nb, final_cluster_nb in enumerate(local_merge_flat_groups.keys()):
            col_nb = 1
            # Prepare median waveforms plot.
            # # Prepare median waveform for each cluster.
            median_traces = {}
            for cluster_nb in local_merge_flat_groups[final_cluster_nb]:
                selection = (allocation == cluster_nb)
                selected_nbs = np.where(selection)[0]
                traces = waveforms_data[selected_nbs]
                median_traces[cluster_nb] = np.median(traces, axis=0)
            # # Prepare median waveform for merged cluster.
            selection = np.in1d(allocation, local_merge_flat_groups[final_cluster_nb])
            selected_nbs = np.where(selection)[0]
            traces = waveforms_data[selected_nbs]
            merged_median_trace = np.median(traces, axis=0)
            # Plot median waveforms.
            ax = axes[row_nb, col_nb]
            ax.set_axis_on()
            for cluster_nb, median_trace in list(median_traces.items()):
                color = colors[cluster_nb]
                ax.plot(median_trace, color=color)
            ax.plot(merged_median_trace, color='black')
            if row_nb == nb_rows - 1:
                ax.set_xlabel("time")
                ax.set_ylabel("amp.")
            if col_nb == 1:
                group = local_merge_groups[final_cluster_nb]

                def group_to_title(g):
                    if isinstance(g, tuple):
                        t = "(" + group_to_title(g[0]) + "+" + group_to_title(g[1]) + ")"
                    else:
                        t = "{:d}".format(g)
                    return t

                title = group_to_title(group)
                if title[0] == '(':
                    title = title[1:-1]
                ax.set_title("cluster {}".format(title))

        for row_nb, final_cluster_nb in enumerate(local_merge_flat_groups.keys()):
            cluster_nbs = local_merge_flat_groups[final_cluster_nb]
            for k, cluster_nb in enumerate(cluster_nbs):
                col_nb = 1 + 1 + k
                # Prepare waveforms plot.
                selection = (allocation == cluster_nb)
                selected_nbs = np.where(selection)[0]
                nb_waveforms = len(selected_nbs)
                selected_nbs = np.random.permutation(selected_nbs)
                selected_nbs = selected_nbs[0:max_nb_traces]
                distance = local_merge_distances[cluster_nb]
                # Plot waveforms.
                ax = axes[row_nb, col_nb]
                ax.set_axis_on()
                axes[row_nb, 1].get_shared_x_axes().join(axes[row_nb, 1], ax)
                axes[row_nb, 1].get_shared_y_axes().join(axes[row_nb, 1], ax)
                for selected_nb in selected_nbs:
                    color_jitter = numpy.random.uniform(low=-0.05, high=0.0)
                    color = colors[cluster_nb]
                    color = tuple([v + color_jitter for v in color])
                    trace = waveforms_data[selected_nb]
                    ax.plot(trace, color=color)
                if distance is None:
                    annotation_text = "n={:d}".format(nb_waveforms)
                else:
                    annotation_text = "n={:d}\nd={:f}".format(nb_waveforms, distance)
                ax.annotate(
                    annotation_text,
                    xy=(0, 0), xycoords='axes fraction',
                    xytext=(2, 1), textcoords='offset points',
                    horizontalalignment='left', verticalalignment='bottom'
                )
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_title("cluster {}".format(cluster_nb))

        fig.tight_layout(rect=[0, 0.05, 1, 1])

        ax.annotate(
            "{} (thr.={:f})".format(merging_method, merging_threshold),
            xy=(0.0, 0.0), xycoords='figure fraction',
            xytext=(10, 2), textcoords='offset points',
            horizontalalignment='left', verticalalignment='bottom'
        )

        if save:
            try:
                output_filename = 'local_merges_%s.%s' % (save[1], save[2])
                output_path = os.path.join(save[0], output_filename)
                plt.savefig(output_path)
                plt.close(fig)
            except Exception:
                pass
            del fig
        else:
            plt.show()

    return


def view_local_merges_backup(
        waveforms_data, clusters_data, old_allocation, new_allocation, merge_history,
        save=False, max_nb_traces=200,
):

    import matplotlib.pyplot as plt

    marker_size = 10
    color_map = plt.get_cmap('jet')

    nb_merges = len(merge_history['merge'])
    allocation = np.copy(old_allocation)

    cluster_nbs = np.unique(old_allocation[old_allocation > - 1])
    colors = {
        cluster_nb: mcolors.to_rgb('C{}'.format(k % 10))
        for k, cluster_nb in enumerate(cluster_nbs)
    }

    for merge_nb in range(0, nb_merges):

        cluster_nb_1, cluster_nb_2 = merge_history['merge'][merge_nb]
        cluster_distance = merge_history['distance'][merge_nb]
        merging_method = merge_history['method']
        merging_threshold = merge_history['threshold']

        fig = plt.figure()

        # Prepare 1st plot.
        selection = (allocation == cluster_nb_1)
        selected_nbs = np.where(selection)[0]
        selected_nbs = np.random.permutation(selected_nbs)
        selected_nbs = selected_nbs[0:max_nb_traces]
        # Plot 1st cluster.
        ax = fig.add_subplot(221)
        for selected_nb in selected_nbs:
            color_jitter = numpy.random.uniform(low=-0.05, high=0.0)
            color = colors[cluster_nb_1]
            color = tuple([v + color_jitter for v in color])
            trace = waveforms_data[selected_nb]
            ax.plot(trace, color=color)
        ax.set_xlabel("time")
        ax.set_ylabel("amp.")
        ax.set_title("cluster {}".format(cluster_nb_1))

        # Prepare 2nd plot.
        selection = (allocation == cluster_nb_2)
        selected_nbs = np.where(selection)[0]
        selected_nbs = np.random.permutation(selected_nbs)
        selected_nbs = selected_nbs[0:max_nb_traces]
        # Plot 2nd cluster.
        ax = fig.add_subplot(222)
        for selected_nb in selected_nbs:
            color_jitter = numpy.random.uniform(low=-0.05, high=0.0)
            color = colors[cluster_nb_2]
            color = tuple([v + color_jitter for v in color])
            trace = waveforms_data[selected_nb]
            ax.plot(trace, color=color)
        ax.set_xlabel("time")
        ax.set_ylabel("amp.")
        ax.set_title(" cluster {}".format(cluster_nb_2))

        # Prepare 3rd plot.
        selection = numpy.logical_or(
            allocation == cluster_nb_1,
            allocation == cluster_nb_2
        )
        selected_nbs = np.where(selection)[0]
        selected_nbs = np.random.permutation(selected_nbs)
        selected_nbs = selected_nbs[0:max_nb_traces]
        # Plot merged cluster.
        ax = fig.add_subplot(223)
        for selected_nb in selected_nbs:
            color_jitter = numpy.random.uniform(low=-0.05, high=0.0)
            color = colors[allocation[selected_nb]]
            color = tuple([v + color_jitter for v in color])
            trace = waveforms_data[selected_nb]
            ax.plot(trace, color=color)
        ax.set_xlabel("time")
        ax.set_ylabel("amp.")
        ax.set_title("cluster {}+{}".format(cluster_nb_1, cluster_nb_2))

        # Prepare 4th plot.
        pca = PCA(n_components=3)
        reduced_clusters_data = pca.fit_transform(clusters_data.astype(numpy.double))
        is_assigned = numpy.where(allocation > -1)[0]
        is_not_assigned = numpy.where(allocation == -1)[0]
        # Plot labelled spike projections.
        ax = fig.add_subplot(224)
        x = reduced_clusters_data[is_not_assigned, 0]
        y = reduced_clusters_data[is_not_assigned, 1]
        ax.scatter(x, y, c='k', linewidth=0, s=marker_size, alpha=0.5)
        x = reduced_clusters_data[is_assigned, 0]
        y = reduced_clusters_data[is_assigned, 1]
        c = np.array([
            colors[cluster_nb]
            for cluster_nb in allocation[is_assigned]
        ])
        ax.scatter(x, y, c=c, cmap=color_map, linewidth=0, s=marker_size)
        ax.set_aspect('equal')
        ax.set_xlabel('dim. 0')
        ax.set_ylabel('dim. 1')
        ax.set_title('2D projection')

        fig.tight_layout(rect=[0, 0.05, 1, 1])

        ax.annotate(
            "{} (thr.={:f})\nd={:f}".format(merging_method, merging_threshold, cluster_distance),
            xy=(0.0, 0.0), xycoords='figure fraction',
            xytext=(10, 2), textcoords='offset points',
            horizontalalignment='left', verticalalignment='bottom'
        )

        if save:
            try:
                output_filename = 'local_merges_%s_%d.%s' % (save[1], merge_nb, save[2])
                output_path = os.path.join(save[0], output_filename)
                plt.savefig(output_path)
                plt.close(fig)
            except Exception:
                pass
            del fig

        # Update `allocations`.
        selection = (allocation == cluster_nb_2)
        allocation[selection] = cluster_nb_1

    if not save:
        plt.show()

    return


def view_rejection(a, b, hist, save=False):

    fig = pylab.figure(figsize=(15, 10))
    ax = fig.add_subplot(211)
    ax.plot(b, a)
    ax.set_xlabel(r'$Amplitude$')
    ax.set_ylabel(r'$Probability$')
    ax.set_title('distribution of amplitudes')

    ax = fig.add_subplot(212)
    ax.plot(b, hist)
    ax.set_xlabel(r'$Amplitude$')
    ax.set_ylabel(r'$Probability$')
    ax.set_title('Rejection curve')

    try:
        pylab.tight_layout()
    except Exception:
        pass
    if save:
        try:
            pylab.savefig(os.path.join(save[0], 'rejection_%s' % save[1]))
            pylab.close()
        except Exception:
            pass
    else:
        pylab.show()
    del fig


def view_waveforms_clusters(data, halo, threshold, templates, amps_lim, n_curves=200, save=False):

    nb_templates = templates.shape[1]
    n_panels = numpy.ceil(numpy.sqrt(nb_templates))
    mask = numpy.where(halo > -1)[0]  # i.e. assigned only
    clust_idx = numpy.unique(halo[mask])
    fig = pylab.figure()
    square = True
    center = len(data[0] - 1) // 2

    my_cmap = pylab.get_cmap('jet')
    cNorm = mcolors.Normalize(vmin=numpy.min(halo), vmax=numpy.max(halo))
    scalarMap = pylab.cm.ScalarMappable(norm=cNorm, cmap=my_cmap)

    for count, i in enumerate(range(nb_templates)):
        if square:
            pylab.subplot(n_panels, n_panels, count + 1)
            if numpy.mod(count, n_panels) != 0:
                pylab.setp(pylab.gca(), yticks=[])
            if count < n_panels*(n_panels - 1):
                pylab.setp(pylab.gca(), xticks=[])

        subcurves = numpy.where(halo == clust_idx[count])[0]
        for k in numpy.random.permutation(subcurves)[:n_curves]:
            colorVal = scalarMap.to_rgba(clust_idx[count])
            pylab.plot(data[k], color=colorVal)

        pylab.plot(templates[:, count], 'r')
        pylab.plot(amps_lim[count][0]*templates[:, count], 'b', alpha=0.5)
        pylab.plot(amps_lim[count][1]*templates[:, count], 'b', alpha=0.5)
        
        xmin, xmax = pylab.xlim()
        pylab.plot([xmin, xmax], [-threshold, -threshold], 'k--')
        pylab.plot([xmin, xmax], [threshold, threshold], 'k--')
        pylab.plot([xmin, xmax], [0, 0], 'k--')
        # pylab.ylim(-1.5*threshold, 1.5*threshold)
        ymin, ymax = pylab.ylim()
        pylab.plot([center, center], [ymin, ymax], 'k--')
        pylab.title('Cluster %d' % i)

    if nb_templates > 0:
        pylab.tight_layout()
    if save:
        pylab.savefig(os.path.join(save[0], 'waveforms_%s' % save[1]))
        pylab.close()
    else:
        pylab.show()
    del fig


def view_artefact(data, save=False):

    fig = pylab.figure()
    pylab.plot(data.T)
    if save:
        pylab.savefig(os.path.join(save[0], 'artefact_%s' % save[1]))
        pylab.close()
    else:
        pylab.show()
    del fig


def view_waveforms(file_name, temp_id, n_spikes=2000):
    """View template next to median snippet."""

    params = CircusParser(file_name)
    data_file = params.get_data_file()
    data_file.open()
    N_e = params.getint('data', 'N_e')
    N_t = params.getint('detection', 'N_t')
    # N_total = params.nb_channels
    sampling_rate = params.rate
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening = params.getboolean('whitening', 'spatial')
    # spike_thresh = params.getfloat('detection', 'spike_thresh')
    # file_out_suff = params.get('data', 'file_out_suff')
    nodes, edges = get_nodes_and_edges(params)
    chunk_size = N_t

    _ = data_file.analyze(chunk_size)  # i.e. count chunks in sources

    spatial_whitening = load_data(params, 'spatial_whitening') if do_spatial_whitening else None
    temporal_whitening = load_data(params, 'temporal_whitening') if do_temporal_whitening else None

    result = load_data(params, 'results')
    spikes = result['spiketimes']['temp_' + str(temp_id)]

    curve = numpy.zeros((n_spikes, N_e, N_t), dtype=numpy.float32)
    templates = load_data(params, 'templates')

    for count, t_spike in enumerate(numpy.random.permutation(spikes)[:n_spikes]):
        t_start = t_spike - int(N_t - 1) // 2
        idx = data_file.get_idx(t_start, chunk_size)
        padding = (0, N_t - 1)
        data, t_offset = data_file.get_data(idx, chunk_size, padding=padding, nodes=nodes)
        data = data[(t_start - t_offset) % chunk_size:(t_start - t_offset) % chunk_size + N_t, :]
        if do_spatial_whitening:
            data = numpy.dot(data, spatial_whitening)
        if do_temporal_whitening:
            data = scipy.ndimage.filters.convolve1d(data, temporal_whitening, axis=0, mode='constant')

        curve[count] = data.T
    data_file.close()

    im1 = numpy.median(curve, axis=0)
    im2 = templates[:, temp_id].toarray().reshape(N_e, N_t)
    imshow_kwargs = {
        'cmap': 'seismic',
        'aspect': 'auto',
        'vmin': - max(np.max(np.abs(im1)), np.max(np.abs(im2))),
        'vmax': + max(np.max(np.abs(im1)), np.max(np.abs(im2))),
        'origin': 'lower',
        'extent': (
            - (N_t - 1) / 2 / sampling_rate * 1e+3,  # left
            + (N_t + 1) / 2 / sampling_rate * 1e+3,  # right
            0.0 - 0.5,  # bottom
            (N_e - 1) + 0.5,  # top
        ),
    }
    axvline_kwargs = {
        'color': 'tab:grey',
        'linewidth': 0.5,
    }
    # Plot template.
    ax = pylab.subplot(121)
    ax.imshow(im2, **imshow_kwargs)
    ax.axvline(**axvline_kwargs)
    ax.set_xlabel("lag (ms)")
    ax.set_ylabel("channel")
    ax.set_title("template (id={})".format(temp_id))
    # Plot median waveform.
    ax = pylab.subplot(122)
    ax.imshow(im1, **imshow_kwargs)
    ax.axvline(**axvline_kwargs)
    ax.set_xlabel("lag (ms)")
    ax.set_ylabel("channel")
    ax.set_title("median snippet (n={})".format(n_spikes))
    pylab.tight_layout()
    pylab.show()

    return curve


def view_isolated_waveforms(file_name, t_start=0, t_stop=1):
    """View isolated peaks."""

    params = CircusParser(file_name)
    data_file = params.get_data_file()
    data_file.open()
    N_e = params.getint('data', 'N_e')
    N_t = params.getint('detection', 'N_t')
    # N_total = params.nb_channels
    sampling_rate = params.rate
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening = params.getboolean('whitening', 'spatial')
    # spike_thresh = params.getfloat('detection', 'spike_thresh')
    # file_out_suff = params.get('data', 'file_out_suff')
    nodes, edges = get_nodes_and_edges(params)
    k_start = int(t_start * sampling_rate)
    k_stop = int(t_stop * sampling_rate)
    chunk_size = k_stop - k_start
    padding = (0, 0)

    _ = data_file.analyze(chunk_size)  # i.e. count chunks in sources

    spatial_whitening = load_data(params, 'spatial_whitening') if do_spatial_whitening else None
    temporal_whitening = load_data(params, 'temporal_whitening') if do_temporal_whitening else None

    thresholds = load_data(params, 'thresholds')
    gidx = data_file.get_idx(k_start, chunk_size)
    data, k_offset = data_file.get_data(gidx, chunk_size, padding=padding, nodes=nodes)
    data = data[(k_start - k_offset):(k_stop - k_offset)]

    peaks = {}
    n_spikes = 0

    if do_spatial_whitening:
        data = numpy.dot(data, spatial_whitening)
    if do_temporal_whitening: 
        for i in range(N_e):
            data[:, i] = numpy.convolve(data[:, i], temporal_whitening, 'same')

    for i in range(N_e):
        peaks[i] = scipy.signal.find_peaks(-data[:, i], height=thresholds[i])[0]
        n_spikes += len(peaks[i])

    curve = numpy.zeros((n_spikes, N_t), dtype=numpy.float32)
    print(("We found %d spikes" % n_spikes))

    count = 0
    for electrode in range(N_e):
        for i in range(len(peaks[electrode])):
            peak_time = peaks[electrode][i]
            if data.shape[0] - N_t / 2 > peak_time > N_t / 2:
                k_start = peak_time - (N_t - 1) // 2
                k_stop = peak_time + (N_t - 1) // 2 + 1
                curve[count] = data[k_start:k_stop, electrode]
            count += 1

    im = curve
    imshow_kwargs = {
        'cmap': 'seismic',
        'aspect': 'auto',
        'vmin': - np.max(np.abs(im)),
        'vmax': + np.max(np.abs(im)),
        'origin': 'lower',
        'extent': (
            - (N_t + 1) / 2 / sampling_rate * 1e+3,  # left
            + (N_t + 1) / 2 / sampling_rate * 1e+3,  # right
            0.0 - 0.5,  # bottom
            float(n_spikes) + 0.5,  # top
        )
    }
    ax = pylab.subplot(111)
    ax.imshow(im, **imshow_kwargs)
    ax.axvline(color='tab:gray', linewidth=0.5)
    ax.set_xlabel("lag")
    ax.set_ylabel("waveform")
    ax.set_title("isolated peaks")
    pylab.tight_layout()
    pylab.show()
    data_file.close()

    return curve


def view_triggers(file_name, triggers, n_elec=2, square=True, xzoom=None, yzoom=None, n_curves=100, temp_id=None):
    
    params = CircusParser(file_name)
    data_file = params.get_data_file()
    data_file.open()
    N_e = params.getint('data', 'N_e')
    N_t = params.getint('detection', 'N_t')
    N_total = params.nb_channels
    sampling_rate = params.rate
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening = params.getboolean('whitening', 'spatial')
    spike_thresh = params.getfloat('detection', 'spike_thresh')
    file_out_suff = params.get('data', 'file_out_suff')
    nodes, edges = get_nodes_and_edges(params)
    chunk_size = N_t

    if temp_id is not None:
        templates = load_data(params, 'templates')
        mytemplate = templates[:, temp_id].toarray().reshape(N_e, N_t)

    if do_spatial_whitening:
        spatial_whitening = load_data(params, 'spatial_whitening')
    if do_temporal_whitening:
        temporal_whitening = load_data(params, 'temporal_whitening')

    thresholds = load_data(params, 'thresholds')    

    curve = numpy.zeros((len(triggers), N_e, N_t), dtype=numpy.float32)
    count = 0

    for count, t_spike in enumerate(triggers):
        padding = (t_spike - N_t / 2, t_spike - N_t / 2)
        data = data_file.get_data(0, N_t, padding=padding, nodes=nodes)
        data_shape = len(data)
        if do_spatial_whitening:
            data = numpy.dot(data, spatial_whitening)
        if do_temporal_whitening:
            data = scipy.ndimage.filters.convolve1d(data, temporal_whitening, axis=0, mode='constant')
        
        curve[count] = data.T

    if not numpy.iterable(n_elec):
        if square:
            idx = numpy.random.permutation(numpy.arange(N_e))[:n_elec ** 2]
        else:
            idx = numpy.random.permutation(numpy.arange(N_e))[:n_elec]
    else:
        idx = n_elec
        n_elec = numpy.sqrt(len(idx))
    pylab.figure()
    data_file.close()

    for count, i in enumerate(idx):
        if square:
            pylab.subplot(n_elec, n_elec, count + 1)
            if numpy.mod(count, n_elec) != 0:
                pylab.setp(pylab.gca(), yticks=[])
            if count < n_elec*(n_elec - 1):
                pylab.setp(pylab.gca(), xticks=[])
        else:
            pylab.subplot(n_elec, 1, count + 1)
            if count != (n_elec - 1):
                pylab.setp(pylab.gca(), xticks=[])
        for k in numpy.random.permutation(numpy.arange(len(curve)))[:n_curves]:
            pylab.plot(curve[k, i, :], '0.25')
        pylab.plot(numpy.mean(curve, 0)[i], 'r')
        xmin, xmax = pylab.xlim()
        pylab.plot([xmin, xmax], [-thresholds[i], -thresholds[i]], 'k--')
        pylab.plot([xmin, xmax], [thresholds[i], thresholds[i]], 'k--')
        if temp_id is not None:
            pylab.plot(mytemplate[i, :], 'b')
        pylab.title('Elec %d' % i)
        if xzoom:
            pylab.xlim(xzoom[0], xzoom[1])
        # pylab.ylim(-5*thresholds[i], 5*thresholds[i])
        if yzoom:
            pylab.ylim(yzoom[0], yzoom[1])
    pylab.tight_layout()
    pylab.show()
    return curve


def view_performance(file_name, triggers, lims=(150,150)):

    params = CircusParser(file_name)
    N_e = params.getint('data', 'N_e')
    N_total = params.getint('data', 'N_total')
    sampling_rate = params.getint('data', 'sampling_rate')
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening = params.getboolean('whitening', 'spatial')
    spike_thresh = params.getfloat('detection', 'spike_thresh')
    file_out_suff = params.get('data', 'file_out_suff')
    N_t = params.getint('detection', 'N_t')
    nodes, edges = get_nodes_and_edges(params)
    chunk_size = N_t

    if do_spatial_whitening:
        spatial_whitening = load_data(params, 'spatial_whitening')
    if do_temporal_whitening:
        temporal_whitening = load_data(params, 'temporal_whitening')

    thresholds = load_data(params, 'thresholds')

    try:
        result = load_data(params, 'results')
    except Exception:
        result = {'spiketimes': {}, 'amplitudes': {}}

    curve = numpy.zeros((len(triggers), len(list(result['spiketimes'].keys())), lims[1] + lims[0]), dtype=numpy.int32)
    count = 0

    for count, t_spike in enumerate(triggers):
        for key in list(result['spiketimes'].keys()):
            elec = int(key.split('_')[1])
            idx = numpy.where((result['spiketimes'][key] > t_spike - lims[0]) & (result['spiketimes'][key] < t_spike + lims[0]))
            curve[count, elec, t_spike - result['spiketimes'][key][idx]] += 1
    pylab.subplot(111)
    pylab.imshow(numpy.mean(curve, 0), aspect='auto') 
    return curve


def view_templates(file_name, temp_id=0, best_elec=None, templates=None):
    """View template."""

    params = CircusParser(file_name)
    _ = params.get_data_file()  # i.e. update N_t
    N_e = params.getint('data', 'N_e')
    # N_total = params.getint('data', 'N_total')
    # sampling_rate = params.getint('data', 'sampling_rate')
    # do_temporal_whitening = params.getboolean('whitening', 'temporal')
    # do_spatial_whitening = params.getboolean('whitening', 'spatial')
    # spike_thresh = params.getfloat('detection', 'spike_thresh')
    # file_out_suff = params.get('data', 'file_out_suff')
    N_t = params.getint('detection', 'N_t')
    nodes, edges = get_nodes_and_edges(params)
    # chunk_size = N_t
    N_total = params.getint('data', 'N_total')
    inv_nodes = numpy.zeros(N_total, dtype=numpy.int32)
    inv_nodes[nodes] = numpy.arange(len(nodes))

    if templates is None:
        templates = load_data(params, 'templates')
        templates = templates.toarray()
        templates = templates.reshape(N_e, N_t, -1)
    clusters = load_data(params, 'clusters')
    probe = params.probe

    positions = []
    for i in list(probe['channel_groups'][1]['geometry'].keys()):
        positions.append(probe['channel_groups'][1]['geometry'][i])
    positions = np.array(positions)
    dx = np.median(np.diff(np.unique(positions[:, 0])))  # horizontal inter-electrode distance
    dy = np.median(np.diff(np.unique(positions[:, 1])))  # vertical inter-electrode distance
    x_scaling = 0.8 * dx / 1.0
    y_scaling = 0.8 * dy / (np.abs(np.min(templates[:, :, temp_id])) + np.abs(np.max(templates[:, :, temp_id])))
    if best_elec is None:
        best_elec = clusters['electrodes'][temp_id]
    elif best_elec == 'auto':
        best_elec = numpy.argmin(numpy.min(templates[:, :, temp_id], 1))
    pylab.figure()
    ax = pylab.gca()
    ax.set_aspect('equal')
    for i in range(N_e):
        x_c, y_c = positions[nodes[i]]
        x = x_scaling * numpy.linspace(-0.5, + 0.5, num=N_t) + x_c
        y = y_scaling * templates[i, :, temp_id] + y_c
        if i == best_elec:
            c = 'tab:red'
        elif i in inv_nodes[edges[nodes[best_elec]]]:
            c = 'black'
        else:
            c = 'tab:gray'
        ax.plot(x, y, color=c)
    pylab.tight_layout()
    pylab.setp(ax, xticks=[], yticks=[])
    pylab.show()

    return best_elec


def view_raw_templates(templates, n_temp=2, square=True):

    N_e, N_t, N_tm = templates.shape
    if not numpy.iterable(n_temp):
        if square:
            idx = numpy.random.permutation(numpy.arange(N_tm // 2))[:n_temp ** 2]
        else:
            idx = numpy.random.permutation(numpy.arange(N_tm // 2))[:n_temp]
    else:
        idx = n_temp

    my_cmap = pylab.get_cmap('winter')
    cNorm = mcolors.Normalize(vmin=0, vmax=N_e)
    scalarMap = pylab.cm.ScalarMappable(norm=cNorm, cmap=my_cmap)

    pylab.figure()
    for count, i in enumerate(idx):
        if square:
            pylab.subplot(n_temp, n_temp, count + 1)
            if numpy.mod(count, n_temp) != 0:
                pylab.setp(pylab.gca(), yticks=[])
            if count < n_temp*(n_temp - 1):
                pylab.setp(pylab.gca(), xticks=[])
        else:
            pylab.subplot(len(idx), 1, count + 1)
            if count != (len(idx) - 1):
                pylab.setp(pylab.gca(), xticks=[])
        for j in range(N_e):
            colorVal = scalarMap.to_rgba(j)
            pylab.plot(templates[j, :, i], color=colorVal)

        pylab.title('Template %d' % i)
    pylab.tight_layout()
    pylab.show()    


def view_whitening(data):
    pylab.subplot(121)
    pylab.imshow(data['spatial'], interpolation='nearest')
    pylab.title('Spatial')
    pylab.xlabel('# Electrode')
    pylab.ylabel('# Electrode')
    pylab.colorbar()
    pylab.subplot(122)
    pylab.title('Temporal')
    pylab.plot(data['temporal'])
    pylab.xlabel('Time [ms]')
    x, y = pylab.xticks()
    pylab.xticks(x, (x - x[-1] // 2) // 10)
    pylab.tight_layout()


def view_masks(file_name, t_start=0, t_stop=1, n_elec=0):

    params = CircusParser(file_name)
    data_file = params.get_data_file()
    data_file.open()
    N_e = params.getint('data', 'N_e')
    N_t = params.getint('detection', 'N_t')
    N_total = params.nb_channels
    sampling_rate = params.rate
    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening = params.getboolean('whitening', 'spatial')
    spike_thresh = params.getfloat('detection', 'spike_thresh')
    file_out_suff = params.get('data', 'file_out_suff')
    nodes, edges = get_nodes_and_edges(params)
    chunk_size = (t_stop - t_start) * sampling_rate
    padding = (t_start * sampling_rate, t_start * sampling_rate)
    inv_nodes = numpy.zeros(N_total, dtype=numpy.int32)
    inv_nodes[nodes] = numpy.argsort(nodes)
    safety_time = params.getint('clustering', 'safety_time')

    if do_spatial_whitening:
        spatial_whitening = load_data(params, 'spatial_whitening')
    if do_temporal_whitening:
        temporal_whitening = load_data(params, 'temporal_whitening')

    thresholds = load_data(params, 'thresholds')
    data = data_file.get_data(0, chunk_size, padding=padding, nodes=nodes)
    data_shape = len(data)
    data_file.close()
    peaks = {}
    indices = inv_nodes[edges[nodes[n_elec]]]

    if do_spatial_whitening:
        data = numpy.dot(data, spatial_whitening)
    if do_temporal_whitening: 
        data = scipy.ndimage.filters.convolve1d(data, temporal_whitening, axis=0, mode='constant')
    
    for i in range(N_e):
        peaks[i] = scipy.signal.find_peaks(-data[:, i], height=thresholds[i])[0]

    pylab.figure()

    for count, i in enumerate(indices):

        pylab.plot(count * 5 + data[:, i], '0.25')
        # xmin, xmax = pylab.xlim()
        pylab.scatter(peaks[i], count * 5 + data[peaks[i], i], s=10, c='r')

    for count, i in enumerate(peaks[n_elec]):
        pylab.axvspan(i - safety_time, i + safety_time, facecolor='r', alpha=0.5)

    pylab.ylim(-5, len(indices) * 5)
    pylab.xlabel('Time [ms]')
    pylab.ylabel('Electrode')
    pylab.tight_layout()
    pylab.setp(pylab.gca(), yticks=[])
    pylab.show()
    return peaks


def view_peaks(file_name, t_start=0, t_stop=1, n_elec=2, square=True, xzoom=None, yzoom=None):
    
    params = CircusParser(file_name)
    data_file = params.get_data_file()
    data_file.open()
    N_e = params.getint('data', 'N_e')
    N_t = params.getint('detection', 'N_t')
    N_total = params.nb_channels
    sampling_rate = params.rate

    do_temporal_whitening = params.getboolean('whitening', 'temporal')
    do_spatial_whitening = params.getboolean('whitening', 'spatial')
    spike_thresh = params.getfloat('detection', 'spike_thresh')
    file_out_suff = params.get('data', 'file_out_suff')
    nodes, edges = get_nodes_and_edges(params)
    chunk_size = np.int64((t_stop - t_start) * sampling_rate)
    padding = (np.int64(t_start * sampling_rate), np.int64(t_start * sampling_rate))

    if do_spatial_whitening:
        spatial_whitening = load_data(params, 'spatial_whitening')
    if do_temporal_whitening:
        temporal_whitening = load_data(params, 'temporal_whitening')

    thresholds = load_data(params, 'thresholds')
    data = data_file.get_data(0, chunk_size, padding=padding, nodes=nodes)
    data_shape = len(data)
    data_file.close()

    peaks      = {}
    
    if do_spatial_whitening:
        data = numpy.dot(data[0], spatial_whitening)
    if do_temporal_whitening: 
        data = scipy.ndimage.filters.convolve1d(data, temporal_whitening, axis=0, mode='constant')
    
    for i in range(N_e):
        peaks[i] = scipy.signal.find_peaks(-data[:, i], height=thresholds[i])[0]

    if not numpy.iterable(n_elec):
        if square:
            idx = numpy.random.permutation(numpy.arange(N_e))[:n_elec ** 2]
        else:
            idx = numpy.random.permutation(numpy.arange(N_e))[:n_elec]
    else:
        idx = n_elec
        n_elec = len(idx)
    pylab.figure()
    for count, i in enumerate(idx):
        if square:
            pylab.subplot(n_elec, n_elec, count + 1)
            if numpy.mod(count, n_elec) != 0:
                pylab.setp(pylab.gca(), yticks=[])
            else:
                pylab.ylabel('Signal')
            if count < n_elec*(n_elec - 1):
                pylab.setp(pylab.gca(), xticks=[])
            else:
                pylab.xlabel('Time [ms]')
        else:
            pylab.subplot(n_elec, 1, count + 1)
            if count != (n_elec - 1):
                pylab.setp(pylab.gca(), xticks=[])
            else:
                pylab.xlabel('Time [ms]')
        pylab.plot(data[:, i], '0.25')
        xmin, xmax = pylab.xlim()
        pylab.scatter(peaks[i], data[peaks[i], i], s=10, c='r')
        pylab.xlim(xmin, xmax)
        pylab.plot([xmin, xmax], [-thresholds[i], -thresholds[i]], 'k--')
        pylab.plot([xmin, xmax], [thresholds[i], thresholds[i]], 'k--')
        pylab.title('Electrode %d' % i)
        if xzoom:
            pylab.xlim(xzoom[0], xzoom[1])
        pylab.ylim(-2 * thresholds[i], +2 * thresholds[i])
        if yzoom:
            pylab.ylim(yzoom[0], yzoom[1])
    pylab.tight_layout()
    pylab.show()
    return peaks


def raster_plot(file_name):

    result = get_results(file_name)
    times = []
    templates = []
    for key in list(result['spiketimes'].keys()):
        template = int(key.split('_')[1])
        times += result['spiketimes'][key].tolist()
        templates += [template]*len(result['spiketimes'][key])
    return numpy.array(times), numpy.array(templates)


def view_norms(file_name, save=True):
    """
    Sanity plot of the norms of the templates.
    
    Arguments:
        file_name : string
        save : boolean
    """

    # Retrieve the key parameters.
    params = CircusParser(file_name)
    norms = load_data(params, 'norm-templates')
    N_tm = norms.shape[0] // 2
    y_margin = 0.1

    # Plot the figure.
    fig, axes = pylab.subplots(2, sharex=True)
    x = numpy.arange(0, N_tm, 1)
    y_cen = norms[0:N_tm]
    y_ort = norms[N_tm:2*N_tm]
    scatter_kwargs = {
        's': 3 ** 2,
        'c': 'black',
    }
    x_min = -1
    x_max = N_tm
    y_cen_dif = numpy.amax(y_cen) - numpy.amin(y_cen)
    y_cen_min = numpy.amin(y_cen) - y_margin * y_cen_dif
    y_cen_max = numpy.amax(y_cen) + y_margin * y_cen_dif
    y_ort_dif = numpy.amax(y_ort) - numpy.amin(y_ort)
    y_ort_min = numpy.amin(y_ort) - y_margin * y_ort_dif
    y_ort_max = numpy.amax(y_ort) + y_margin * y_ort_dif
    ax = axes[0]
    ax.scatter(x, y_cen, **scatter_kwargs)
    ax.grid()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_cen_min, y_cen_max])
    ax.set_title("Norms of the {} templates in {}".format(N_tm, file_name.replace('_', '\_')))
    ax.set_xlabel("template (central component)")
    ax.set_ylabel("norm")
    ax = axes[1]
    ax.scatter(x, y_ort, **scatter_kwargs)
    ax.grid()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim([y_ort_min, y_ort_max])
    ax.set_xlabel("template (orthogonal component)")
    ax.set_ylabel("norm")

    # Display the figure.
    if save:
        fig.savefig("/tmp/norms-templates.pdf")
        pylab.close(fig)
    else:
        fig.show()

    return


def view_triggers_bis(file_name, mode='random', save=True):
    """
    Sanity plot of the triggers of a given dataset.
    
    Arguments:
        file_name : string
        mode: string
        save : boolean
    """
    
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import scipy as sp

    # Set global settings.
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.titlesize'] = plt.rcParams['font.size'] + 2
    plt.rcParams['axes.labelsize'] = plt.rcParams['font.size'] - 2
    plt.rcParams['axes.titlesize'] = plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size'] - 2
    plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size'] - 2
    plt.rcParams['axes.linewidth'] = 1
    
    # Retrieve the key parameters.
    params = CircusParser(file_name)
    triggers, spikes = load_data(params, 'triggers')
    
    mean_spike = numpy.mean(spikes, axis=2)
    
    # TODO: remove print zone
    print("# best_elec")
    
    K = mean_spike.shape[1]
    wf_ind = numpy.arange(0, K)
    wf_dif = numpy.zeros(K)
    for k in range(0, K):
        wf = mean_spike[:, k]
        wf_min = numpy.amin(wf)
        wf_max = numpy.amax(wf)
        wf_dif[k] = wf_max - wf_min
    wf_agm = numpy.argsort(wf_dif)
    #
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.suptitle("Best elec (%d, %d, %d, ...)" %(wf_agm[-1], wf_agm[-2], wf_agm[-3]))
    ax = fig.gca()
    ax.plot(wf_ind, wf_dif, 'o')
    ax.grid()
    plt.savefig("/tmp/best-elec.png")
    #
    print((mean_spike.shape))
    # end print zone

    mean_norm = numpy.linalg.norm(mean_spike)
    spikes_bis = spikes.reshape(spikes.shape[0] * spikes.shape[1], spikes.shape[2])
    mean_spike_bis = mean_spike.reshape(mean_spike.shape[0] * mean_spike.shape[1], 1)
    mean_spike_bis = mean_spike_bis[::-1, :]
    spike_amplitudes = (1.0 / (mean_norm ** 2)) * sp.signal.convolve(spikes_bis, mean_spike_bis, mode='valid').flatten()

    N_tr = triggers.shape[0]
    N = min(N_tr, 15)
    if 'random' == mode:
        numpy.random.seed(seed=0)
        idxs = numpy.random.choice(N_tr, size=N, replace=False)
        idxs = numpy.sort(idxs)
    elif 'minimal' == mode:
        idxs = numpy.argsort(spike_amplitudes)
        idxs = idxs[:N]
        # idxs = numpy.sort(idxs)
    elif 'maximal' == mode:
        idxs = numpy.argsort(spike_amplitudes)
        idxs = idxs[-N:]
        # idxs = numpy.sort(idxs)
    
    v_min = min(numpy.amin(spikes[:, :, idxs]), numpy.amax(spikes[:, :, idxs]))
    v_max = - v_min

    # Plot the figure.
    fig = plt.figure()
    gs = gridspec.GridSpec(4, 4)
    fig.suptitle("Ground truth triggers from `%s`" % file_name)
    for (k, ss) in enumerate(gs):
        ax = fig.add_subplot(ss)
        if 0 == k:
            ax.imshow(mean_spike.T, cmap='seismic', interpolation='nearest',
                      vmin=v_min, vmax=v_max)
            ax.set_title("mean spike")
        else:
            idx = idxs[k-1]
            ax.imshow(spikes[:, :, idx].T, cmap='seismic', interpolation='nearest',
                      vmin=v_min, vmax=v_max)
            ax.set_title("spike %d (%f)" % (idx, spike_amplitudes[idx]))
    gs.tight_layout(fig, pad=0.5, h_pad=0.5, w_pad=0.5, rect=[0.0, 0.0, 1.0, 0.95])
    
    xmin = -1
    xmax = numpy.amax(triggers) + 1
    ydiff = numpy.amax(spike_amplitudes) - numpy.amin(spike_amplitudes)
    ymin = min(0.0, numpy.amin(spike_amplitudes)) - 0.1 * ydiff
    ymax = max(0.0, numpy.amax(spike_amplitudes)) + 0.1 * ydiff
    
    # Plot the second figure.
    fig2 = plt.figure()
    gs = gridspec.GridSpec(1, 1)
    fig2.suptitle("Ground truth triggers from `%s`" % file_name)
    ax = fig2.add_subplot(gs[0])
    ax.plot(triggers, spike_amplitudes, 'o')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.grid()

    plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
    weights = (1.0 / spike_amplitudes.shape[0]) * numpy.ones(spike_amplitudes.shape[0])
    q75, q25 = numpy.percentile(spike_amplitudes, [75, 25])
    iqr = q75 - q25
    h = 2.0 * iqr * float(spike_amplitudes.shape[0]) ** (- 1.0 / 3.0)
    bins = int(numpy.amax(spike_amplitudes) - numpy.amin(spike_amplitudes) / h)

    # Plot the third figure.
    fig3 = plt.figure()
    gs = gridspec.GridSpec(1, 1)
    fig3.suptitle("Ground truth triggers from `%s`" % file_name)
    ax = fig3.add_subplot(gs[0])
    ax.hist(spike_amplitudes, bins=bins, weights=weights)
    ax.grid()
    ax.set_xlabel("Amplitudes")
    ax.set_ylabel("Probability")
    
    # Display the figure.
    if save:
        fig.savefig("/tmp/triggers-" + mode + ".png")
        fig2.savefig("/tmp/triggers-amplitudes.png")
        fig3.savefig("/tmp/triggers-amplitudes-hist.png")
        pylab.close(fig)
        pylab.close(fig2)
    else:
        fig.show()
        fig2.show()
        fig3.show()
    
    return


# Validation plots.

def view_trigger_snippets_bis(trigger_snippets, elec_index, save=None):
    fig = pylab.figure()
    ax = fig.add_subplot(1, 1, 1)
    for n in range(0, trigger_snippets.shape[2]):
        y = trigger_snippets[:, elec_index, n]
        x = numpy.arange(- (y.size - 1) / 2, (y.size - 1) / 2 + 1)
        b = 0.5 + 0.5 * numpy.random.rand()
        ax.plot(x, y, color=(0.0, 0.0, b), linestyle='solid')
    ax.grid(True)
    ax.set_xlim([numpy.amin(x), numpy.amax(x)])
    ax.set_xlabel("time")
    ax.set_ylabel("amplitude")
    if save is None:
        pylab.show()
    else:
        pylab.savefig(save)
        pylab.close(fig)
    return


def view_trigger_snippets(trigger_snippets, chans, save=None):
    # Create output directory if necessary.
    if os.path.exists(save):
        for f in os.listdir(save):
            p = os.path.join(save, f)
            os.remove(p)
        os.removedirs(save)
    os.makedirs(save)
    # Plot figures.
    fig = pylab.figure()
    for (c, chan) in enumerate(chans):
        ax = fig.add_subplot(1, 1, 1)
        for n in range(0, trigger_snippets.shape[2]):
            y = trigger_snippets[:, c, n]
            x = numpy.arange(- (y.size - 1) / 2, (y.size - 1) / 2 + 1)
            b = 0.5 + 0.5 * numpy.random.rand()
            ax.plot(x, y, color=(0.0, 0.0, b), linestyle='solid')
        y = numpy.mean(trigger_snippets[:, c, :], axis=1)
        x = numpy.arange(- (y.size - 1) / 2, (y.size - 1) / 2 + 1)
        ax.plot(x, y, color=(1.0, 0.0, 0.0), linestyle='solid')
        ax.grid(True)
        ax.set_xlim([numpy.amin(x), numpy.amax(x)])
        ax.set_title("Channel %d" % chan)
        ax.set_xlabel("time")
        ax.set_ylabel("amplitude")
        if save is not None:
            # Save plot.
            filename = "channel-%d.png" % chan
            path = os.path.join(save, filename)
            pylab.savefig(path)
        fig.clf()
    if save is None:
        pylab.show()
    else:
        pylab.close(fig)
    return


def view_trigger_times(params, spike_times_juxta, juxta_spikes, juxta_spikes_=None, save=None):

    fig = pylab.figure()
    pylab.subplots_adjust(wspace=0.3)

    ax = pylab.subplot2grid((2, 2), (0, 0), rowspan=2)
    sampling_rate = params.getint('data', 'sampling_rate')
    isis = numpy.diff(spike_times_juxta)*1000/sampling_rate
    x, y = numpy.histogram(isis, bins=numpy.linspace(0, 100, 50))
    ax.plot(y[1:], x/float(x.sum()))
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Probability')
    ax.set_title('ISI')

    # TODO: clean temporary zone
    selected_juxta_indices = numpy.random.permutation(juxta_spikes.shape[1])[:1000]
    # selected_juxta_indices = numpy.arange(0, juxta_spikes.shape[1])
    # end temporary zone

    if juxta_spikes_ is not None:
        pad = 0.1
        xmin = 0
        xmax = juxta_spikes_.shape[0] - 1
        if len(selected_juxta_indices) > 0:
            ymin = numpy.amin(juxta_spikes_[:, selected_juxta_indices])
            ymax = numpy.amax(juxta_spikes_[:, selected_juxta_indices])
        else:
            ymin, ymax = 0, 1
        ydif = ymax - ymin
        ymin -= pad * ydif
        ymax += pad * ydif
        ax = pylab.subplot2grid((2, 2), (0, 1))
        for i in selected_juxta_indices:
            ax.plot(juxta_spikes_[:, i], '0.5')
        ax.plot(numpy.mean(juxta_spikes_, axis=1), '0.25')
        ax.plot(2 * [juxta_spikes_.shape[0] / 2], [ymin, ymax], '0.25')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Amplitude [au]')
        ax.set_title('Juxta STA')
    
    pad = 0.1
    xmin = 0
    xmax = juxta_spikes.shape[0] - 1
    if len(selected_juxta_indices) > 0:
        ymin = numpy.amin(juxta_spikes[:, selected_juxta_indices])
        ymax = numpy.amax(juxta_spikes[:, selected_juxta_indices])
    else:
        ymin, ymax = 0, 1
    ydif = ymax - ymin
    ymin -= pad * ydif
    ymax += pad * ydif
    if juxta_spikes_ is None:
        ax = pylab.subplot2grid((2, 2), (0, 1), rowspan=2)
    else:
        ax = pylab.subplot2grid((2, 2), (1, 1))
    for i in selected_juxta_indices:
        ax.plot(juxta_spikes[:, i], '0.5')
    ax.plot(numpy.mean(juxta_spikes, axis=1), '0.25')
    ax.plot(2 * [juxta_spikes.shape[0] / 2], [ymin, ymax], '0.25')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Amplitude [au]')
    ax.set_title('Extra STA')
    
    fig.tight_layout()
    
    if save is None:
        pylab.show()
    else:
        pylab.savefig(save)
        pylab.close(fig)
    return


def view_dataset(X, color='blue', title=None, save=None):
    n_components = 2
    pca = PCA(n_components)
    pca.fit(X)
    x = pca.transform(X)
    fig = pylab.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x[:, 0], x[:, 1], c=color, s=5, lw=0.1)
    ax.grid(True)
    if title is None:
        ax.set_title("Dataset ({} samples)".format(X.shape[0]))
    else:
        ax.set_title(title + " ({} samples)".format(X.shape[0]))
    ax.set_xlabel("1st component")
    ax.set_ylabel("2nd component")
    if save is None:
        pylab.show()
    else:
        pylab.savefig(save)
        pylab.close(fig)
    return


def view_datasets(params, xs, ys, all_trigger_times, colors=None, labels=None, save=None):
    if colors is None:
        colors = ['b'] * len(xs)
    from circus.validating.utils import Projection, find_rotation
    p = Projection()
    p = p.fit(xs, ys)
    x = p.transform(xs)
    pad = 0.05
    x_dif = numpy.amax(x[:, 0]) - numpy.amin(x[:, 0])
    x_min = numpy.amin(x[:, 0]) - pad * x_dif
    x_max = numpy.amax(x[:, 0]) + pad * x_dif
    y_dif = numpy.amax(x[:, 1]) - numpy.amin(x[:, 1])
    y_min = numpy.amin(x[:, 1]) - pad * y_dif
    y_max = numpy.amax(x[:, 1]) + pad * y_dif
    fig = pylab.figure()
    ax = fig.add_subplot(1, 2, 1)
    k = 0
    handles = []
    for (i, X) in enumerate(xs):
        l = X.shape[0]
        if labels is None:
            ax.scatter(x[k:k+l, 0], x[k:k+l, 1], c=colors[i], s=5, lw=0.1)
        else:
            sc = ax.scatter(x[k:k+l, 0], x[k:k+l, 1], c=colors[i], s=5, lw=0.1, label=labels[i])
            handles.append(sc)
        k = k + l
    ax.grid(True)
    # ax.set_aspect('equal')
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_title("Datasets")
    ax.set_xlabel("1st component")
    ax.set_ylabel("2nd component")
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.15,
    #                  box.width, box.height * 0.85])
    # handles = [handles[2], handles[0], handles[1]]
    # labels = [labels[2], labels[0], labels[1]]
    # ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=False, shadow=False, ncol=3)

    N_total = params.nb_channels
    ttmax = params.data_file.duration

    pylab.subplots_adjust(wspace=0.3)
    ax = fig.add_subplot(1, 2, 2)
    sizes = []

    for count, trigger_times in enumerate(all_trigger_times):
        x = numpy.concatenate((
            numpy.array([0]),
            trigger_times,
            numpy.array([ttmax - 1]),
        ))
        x = x.astype('float') * 100.0 / float(ttmax - 1)
        sizes += [x.size]
        y = numpy.linspace(0.0, 100.0, x.size)
        ax.step(x, y, color=colors[count], linestyle='solid', where='post')
    ax.grid(True)
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, 100.0)
    ax.legend(('GT', 'Non GT', 'Noise'), loc='best')
    ax.plot([0.0, 100.0], [0.0, 100.0], color='black', linestyle='dashed')
    # ax.set_aspect('equal')
    ax.set_title("Empirical distribution of triggers")
    ax.set_xlabel("cumulative share of samples (in %)")
    ax.set_ylabel("cumulative share of triggers (in %)")
    if save is None:
        pylab.show()
    else:
        pylab.savefig(save)
        pylab.close(fig)
    return


def view_accuracy(data1, data2, title=None, save=None):
    """Plot accuracy curve"""

    cutoffs, accs, cutoff, acc = data1

    fig = pylab.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(cutoffs, accs, color='blue', linestyle='solid')
    ax.plot(cutoff, acc, color='blue', marker='o')
    ax.grid(True)
    ax.set_xlim([numpy.amin(cutoffs), numpy.amax(cutoffs)])
    ax.set_ylim([0.0, 1.0])
    ax.set_title("Accuracy curve")
    ax.set_xlabel("cutoff")
    ax.set_ylabel("accuracy")
    # Save accuracy plot.

    cutoffs, tprs, tnrs, norm_accs, cutoff, norm_acc = data2
    labels = [
        "true positive rate",
        "true negative rate",
        "normalized accuracy",
    ]

    ax = fig.add_subplot(1, 2, 2)
    h1, = ax.plot(cutoffs, tprs, color='green', linestyle='solid', label=labels[0])
    h2, = ax.plot(cutoffs, tnrs, color='red', linestyle='solid', label=labels[1])
    h3, = ax.plot(cutoffs, norm_accs, color='blue', linestyle='solid', label=labels[2])
    ax.plot(cutoff, norm_acc, color='blue', marker='o')
    ax.grid(True)
    ax.set_xlim([numpy.amin(cutoffs), numpy.amax(cutoffs)])
    ax.set_ylim([0.0, 1.0])
    ax.set_title("Normalized accuracy curve")
    ax.set_xlabel("cutoff")
    ax.set_ylabel("")
    ax.legend([h1, h2, h3], labels)

    if save is None:
        pylab.show()
    else:
        pylab.savefig(save)
        pylab.close(fig)

    return


def view_classifier(params, data_1, data_2, save=None, verbose=False):
    """Plot classifier"""
    # Retrieve parameters.

    from circus.validating.utils import Projection, find_rotation, find_apparent_contour

    fig = pylab.figure()
    for count, item in enumerate([data_1, data_2]):
        X, y, A, b, c = item
        ax = fig.add_subplot(1, 2, count+1)
        p = Projection()
        p = p.fit(X, y)
        if len(X) == 3:
            X_gt, X_ngt, X_noi = X
            y_gt, y_ngt, y_noi = y
        elif len(X) == 2:
            X_gt, X_ngt = X
            y_gt, y_ngt = y
        X_raw = numpy.vstack(tuple(X))
        # Data transformation.
        X_raw_ = p.transform(X_raw)
        X_gt_ = p.transform(X_gt)
        X_ngt_ = p.transform(X_ngt)
        if len(X) == 3:
            X_noi_ = p.transform(X_noi)
        # Means transformation.
        mu_gt = numpy.mean(X_gt, axis=0).reshape(1, -1)
        mu_gt_ = p.transform(mu_gt)
        mu_ngt = numpy.mean(X_ngt, axis=0).reshape(1, -1)
        mu_ngt_ = p.transform(mu_ngt)
        if len(X) == 3:
            mu_noi = numpy.mean(X_noi, axis=0).reshape(1, -1)
            mu_noi_ = p.transform(mu_noi)
        # Ellipse transformation.
        f = 0.25 * numpy.dot(numpy.dot(b, numpy.linalg.inv(A)), b) - c
        t = - 0.5 * numpy.dot(numpy.linalg.inv(A), b).reshape(1, -1)
        s, O = numpy.linalg.eigh(numpy.linalg.inv((1.0 / f) * A))
        # TODO: remove following line if possible.
        s = numpy.abs(s)
        s = numpy.sqrt(s)
        t_ = p.transform(t)
        O_ = p.transform(numpy.multiply(O, s).T + t)
        if verbose:
            # msg = [
            #     "# s (i.e. demi-axes)",
            #     "%s" %(s,),
            # ]
            # io.print_and_log(msg, level='default', logger=params)
            pass
        # Find plot limits.
        pad = 0.3
        x_dif = numpy.amax(X_raw_[:, 0]) - numpy.amin(X_raw_[:, 0])
        x_min = numpy.amin(X_raw_[:, 0]) - pad * x_dif
        x_max = numpy.amax(X_raw_[:, 0]) + pad * x_dif
        y_dif = numpy.amax(X_raw_[:, 1]) - numpy.amin(X_raw_[:, 1])
        y_min = numpy.amin(X_raw_[:, 1]) - pad * y_dif
        y_max = numpy.amax(X_raw_[:, 1]) + pad * y_dif
        # Retrieve the projection vectors.
        v1, v2 = p.get_vectors()
        if verbose:
            # msg = [
            #     "# norm(v1)",
            #     "%s" %(numpy.linalg.norm(v1),),
            #     "# norm(v2)",
            #     "%s" %(numpy.linalg.norm(v2),),
            # ]
            # io.print_and_log(msg, level='default', logger=params)
            pass
        # Find a rotation which maps theses vectors on the two first vectors of the
        # canonical basis of R^m.
        R = find_rotation(v1, v2)
        # Apply rotation to the classifier.
        R_ = R.T
        mean_ = p.get_mean()
        A_ = numpy.dot(numpy.dot(R_.T, A), R_)
        b_ = numpy.dot(R_.T, 2.0 * numpy.dot(A, mean_) + b)
        c_ = numpy.dot(numpy.dot(A, mean_) + b, mean_) + c
        if verbose:
            # msg = [
            #     "# mean_",
            #     "%s" %(mean_,),
            # ]
            # io.print_and_log(msg, level='default', logger=params)
            pass
        # Find the apparent contour of the classifier.
        A__, b__, c__ = find_apparent_contour(A_, b_, c_)
        # Plot classifier.
        
        # # Plot datasets.
        ax.scatter(X_ngt_[:, 0], X_ngt_[:, 1], c='b', s=5, lw=0.1)
        if len(X) == 3:
            ax.scatter(X_noi_[:, 0], X_noi_[:, 1], c='k', s=5, lw=0.1)
        ax.scatter(X_gt_[:, 0], X_gt_[:, 1], c='r', s=5, lw=0.1)
        # # Plot ellipse transformation.
        for i in range(0, O_.shape[0]):
            ax.plot([t_[0, 0], O_[i, 0]], [t_[0, 1], O_[i, 1]], 'y', zorder=3)
        # # Plot ellipse apparent contour.
        n = 300
        x_r = numpy.linspace(x_min, x_max, n)
        y_r = numpy.linspace(y_min, y_max, n)
        xx, yy = numpy.meshgrid(x_r, y_r)
        zz = numpy.zeros(xx.shape)
        for i in range(0, xx.shape[0]):
            for j in range(0, xx.shape[1]):
                v = numpy.array([xx[i, j], yy[i, j]])
                zz[i, j] = numpy.dot(numpy.dot(v, A__), v) + numpy.dot(b__, v) + c__
        vv = numpy.array([0.0])
        # vv = numpy.arange(0.0, 1.0, 0.1)
        # vv = numpy.arange(0.0, 20.0)
        ax.contour(xx, yy, zz, vv, colors='y', linewidths=1.0)
        # cs = ax.contour(xx, yy, zz, vv, colors='k', linewidths=1.0)
        # ax.clabel(cs, inline=1, fontsize=10)
        # # Plot means of datasets.
        ax.scatter(mu_gt_[:, 0], mu_gt_[:, 1], c='y', s=30, lw=0.1, zorder=4)
        ax.scatter(mu_ngt_[:, 0], mu_ngt_[:, 1], c='y', s=30, lw=0.1, zorder=4)
        if len(X) == 3:
            ax.scatter(mu_noi_[:, 0], mu_noi_[:, 1], c='y', s=30, lw=0.1, zorder=4)
        # # Plot aspect.
        # ax.set_aspect('equal')
        ax.grid()
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])

        if count == 0:
            ax.set_title("Before")
            ax.set_xlabel("1st component")
            ax.set_ylabel("2nd component")
        else:
            ax.set_title("After")
            ax.set_xlabel("1st component")

    if save is None:
        pylab.show()
    else:
        pylab.savefig(save)
        pylab.close(fig)

    return


def view_mahalanobis_distribution(data_1, data_2, save=None):
    """Plot Mahalanobis distribution Before and After"""

    fig = pylab.figure()
    ax = fig.add_subplot(1, 2, 1)
    if len(data_1) == 3:
        d_gt, d_ngt, d_noi = data_1
    elif len(data_1) == 2:
        d_gt, d_ngt = data_1
    if len(data_1) == 3:
        ax.hist(d_noi, bins=50, color='k', alpha=0.5, label="Noise")
    ax.hist(d_ngt, bins=50, color='b', alpha=0.5, label="Non GT")
    ax.hist(d_gt, bins=75, color='r', alpha=0.5, label="GT")
    ax.grid(True)
    ax.set_title("Before")
    ax.set_ylabel("")
    ax.set_xlabel('# Samples')
    ax.set_xlabel('Distances')
    
    if len(data_2) == 3:
        d_gt, d_ngt, d_noi = data_2
    elif len(data_2) == 2:
        d_gt, d_ngt = data_2
    ax = fig.add_subplot(1, 2, 2)
    if len(data_2) == 3:
        ax.hist(d_noi, bins=50, color='k', alpha=0.5, label="Noise")
    ax.hist(d_ngt, bins=50, color='b', alpha=0.5, label="Non GT")
    ax.hist(d_gt, bins=75, color='r', alpha=0.5, label="GT")
    ax.grid(True)
    ax.set_title("After")
    ax.set_ylabel("")
    ax.set_xlabel('Distances')
    
    ax.legend()
    if save is None:
        pylab.show()
    else:
        pylab.savefig(save)
        pylab.close(fig)

    return


def view_classification(data_1, data_2, title=None, save=None):

    fig = pylab.figure()
    count = 0
    panels = [0, 2, 1, 3]
    for item in [data_1, data_2]:
        clf, cld, X, X_raw, y = item
        for mode in ['predict', 'decision_function']:
            ax = fig.add_subplot(2, 2, panels[count]+1)

            if mode == 'predict':
                c = clf
                vmax = 1.0
                vmin = 0.0
            elif mode == 'decision_function':
                c = cld
                vmax = max(abs(numpy.amin(c)), abs(numpy.amax(c)))
                vmin = - vmax

            from circus.validating.utils import Projection
            p = Projection()
            _ = p.fit(X_raw, y)
            X_raw_ = p.transform(X_raw)
            # Plot figure.
            sc = ax.scatter(X_raw_[:, 0], X_raw_[:, 1], c=c, s=5, lw=0.1, cmap='bwr',
                            vmin=vmin, vmax=vmax)
            cb = fig.colorbar(sc)
            ax.grid(True)
            if panels[count] in [0, 1]:
                if panels[count] == 0:
                    ax.set_title('Classification Before')
                    ax.set_ylabel("2nd component")
                if panels[count] == 1:
                    ax.set_title('Classification After')
                    cb.set_label('Prediction')
            elif panels[count] in [2, 3]:
                ax.set_xlabel("1st component")
                if panels[count] == 2:
                    ax.set_ylabel("2nd component")
                if panels[count] == 3:
                    cb.set_label('Decision function')
            count += 1

    if save is None:
        pylab.show()
    else:
        pylab.savefig(save)
        pylab.close(fig)

    return


def view_loss_curve(losses, title=None, save=None):
    """Plot loss curve"""
    x_min = 1
    x_max = len(losses) - 1
    fig = pylab.figure()
    ax = fig.gca()
    ax.semilogy(list(range(x_min, x_max + 1)), losses[1:], color='blue', linestyle='solid')
    ax.grid(True, which='both')
    if title is None:
        ax.set_title("Loss curve")
    else:
        ax.set_title(title)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_xlim([x_min - 1, x_max + 1])
    if save is None:
        pylab.show()
    else:
        pylab.savefig(save)
        pylab.close(fig)
    return


def view_roc_curve(params, fprs, tprs, fpr, tpr, scerror=None, save=None):
    """Plot ROC curve"""

    fig = pylab.figure()
    pylab.subplots_adjust(wspace=0.3)

    HAVE_RESULT = True

    if HAVE_RESULT:
        ax = fig.add_subplot(121)
    else:
        ax = fig.add_subplot(111)

    ax.plot([0.0, 1.0], [0.0, 1.0], color='black', linestyle='dashed')
    ax.plot(fprs, tprs, color='blue', linestyle='solid', zorder=3)
    if fpr is not None and tpr is not None:
        ax.plot(fpr, tpr, color='blue', marker='o', zorder=4)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_title("ROC curve")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")

    def get_fprs(confusion_matrices):
        """Get false positive rates"""
        # Compute false positive rates.
        fprs = [M[1, 0] / (M[1, 0] + M[1, 1]) for M in confusion_matrices]
        # Add false positive rate endpoints.
        fprs = [1.0] + fprs + [0.0]
        return fprs

    def get_tprs(confusion_matrices):
        """Get true positive rates"""
        # Compute true positive rates.
        tprs = [M[0, 0] / (M[0, 0] + M[0, 1]) for M in confusion_matrices]
        # Add true positive rate endpoints.
        tprs = [1.0] + tprs + [0.0]
        return tprs

    def get_fpers(confusion_matrices):
        """Get false positive error rates"""
        # Compute false positive error rates.
        fpers = [M[1, 0] / (M[0, 0] + M[1, 0]) for M in confusion_matrices]
        # Add false positive error rate endpoints.
        fpers = [1.0] + fpers + [0.0]
        return fpers

    def get_fners(confusion_matrices):
        """ Get false negative error rates"""
        # Compute false negative error rates.
        fners = [M[0, 1] / (M[0, 0] + M[0, 1]) for M in confusion_matrices]
        # Add false negative error rate endpoints.
        fners = [0.0] + fners + [1.0]
        return fners

    if HAVE_RESULT:

        ax = fig.add_subplot(122)

        # Retrieve the confusion matrices.
        confusion_matrices = load_data(params, "confusion-matrices")

        if scerror is None:

            # Parameters needed to compute scerror:
            # params
            # File variables needed to compute scerror:
            # juxta-triggers, sampling-rate, matching-jitter, results, templates

            # Retrieve the juxtacellular spike times.
            all_times = load_data(params, "juxta-triggers")

            thresh = int(params.getint('data', 'sampling_rate')*2*1e-3)
            # print("Time difference threshold: {}".format(thresh))

            # Retrieve the SpyKING CIRCUS spike times.
            result = load_data(params, "results")
            data = result['spiketimes']

            # Retrieve the templates.
            templates = load_data(params, 'templates')

            n_temp = len(data)
            res = numpy.zeros((n_temp, 2))
            rates = []
            nb_total = len(all_times)
            nb_fitted = 0

            # Count the number of spike times sorted by SpyKING CIRCUS.
            for i in range(n_temp):
                nb_fitted += len(data['temp_' + str(i)])

            print(("Number of spikes {}/{} with {} templates".format(nb_fitted, nb_total, n_temp)))

            # # First pass to detect what are the scores.
            for i in range(n_temp):
                spikes = data['temp_' + str(i)]
                # print "Template", i, "with", len(spikes), "spikes"
                # Compute the false positive rate.
                for spike in all_times:
                    idx = numpy.where(abs(spikes - spike) <= thresh)[0]
                    if len(idx) > 0:
                        res[i, 0] += 1
                if len(all_times) > 0:
                    res[i, 0] /= float(len(all_times))
                # Compute the positive predictive value.
                for spike in spikes:
                    idx = numpy.where(abs(all_times - spike) <= thresh)[0]
                    if len(idx) > 0:
                        res[i, 1] += 1
                if len(spikes) > 0:
                    res[i, 1] /= float(len(spikes))

            idx = numpy.argmax(numpy.mean(res, 1))
            selection = [idx]
            error = res[idx]
            find_next = True
            source_temp = templates[:, idx].toarray().flatten()
            temp_match = []
            dmax = 0.1
            for i in range(templates.shape[1]/2):
                d = numpy.corrcoef(templates[:, i].toarray().flatten(), source_temp)[0, 1]
                if d > dmax and i not in selection:
                    temp_match += [i]

            # # Second pass to reach the best score with greedy aggregations.
            if 0 < len(temp_match):

                while find_next:

                    temp_match = [i for i in temp_match if i not in selection]

                    local_errors = numpy.zeros((len(temp_match), 2))

                    for mcount, tmp in enumerate(temp_match):

                        # Gather selected spikes.
                        spikes = []
                        for xtmp in selection + [tmp]:
                            spikes += data['temp_' + str(xtmp)].tolist()
                        spikes = numpy.array(spikes, dtype=numpy.int32)

                        # Compute true positive rate.
                        count = 0
                        for spike in all_times:
                            idx = numpy.where(numpy.abs(spikes - spike) < thresh)[0]
                            if len(idx) > 0:
                                count += 1
                        if len(all_times) > 0:
                            local_errors[mcount, 0] = count / float(len(all_times))

                        # Compute positive predictive value
                        count = 0
                        for spike in spikes:
                            idx = numpy.where(numpy.abs(all_times - spike) < thresh)[0]
                            if len(idx) > 0:
                                count += 1
                        if len(spikes) > 0:
                            local_errors[mcount, 1] = count / (float(len(spikes)))
                    
                    errors = numpy.mean(local_errors, 1)
                    if numpy.max(errors) > numpy.mean(error):
                        idx = numpy.argmax(errors)
                        selection += [temp_match[idx]]
                        error = local_errors[idx]
                    else:
                        find_next = False

            error = 100 * (1 - error)
            res = 100 * (1 - res)

        else:

            # Retrieve saved errors for SpyKING CIRCUS.
            res = scerror['res']
            selection = scerror['selection']
            error = scerror['error']

        print(("Best error is obtained with templates {} : {}".format(selection, error)))

        # TODO clean quarantine zone
        # ## Finally, we compute the ROC curve.
        # fprs = get_fprs(confusion_matrices)
        # tprs = get_tprs(confusion_matrices)
        # ## And scale convert it in percent.
        # fprs = [100.0 * fpr for fpr in fprs]
        # tprs = [100.0 * tpr for tpr in tprs]
        
        # # Finally, we compute the performance curve.
        fpers = get_fpers(confusion_matrices)
        fners = get_fners(confusion_matrices)
        # # And scale convert it in percent.
        fpers = [100.0 * fper for fper in fpers]
        fners = [100.0 * fner for fner in fners]
        # TODO end quarantine zone
        
        # TODO clean quarantine zone
        anot_size = 8
        # # Plot the performances of each templates.
        # TODO: check which is the fpr and which is the tpr
        # scatter(res[:, 0], res[:, 1])
        ax.scatter(res[:, 1], res[:, 0])
        for i in range(res.shape[0]):
            txt = str(i)
            # pos = (res[i, 0], res[i, 1])
            pos = (res[i, 1], res[i, 0])
            ax.annotate(txt, pos, horizontalalignment=True, verticalalignment=True, size=anot_size)
        # # Plot the performances of the best aggregations of templates.
        ax.scatter(error[1], error[0])
        pos = (error[1], error[0])
        ax.annotate("best", pos, horizontalalignment=True, verticalalignment=True, size=anot_size)
        # # Plot the performances of the BEER.
        ax.plot(fpers, fners)
        # ax.scatter(fpers, fners, color='r')
        # # Enhance figure.
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.grid(True)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_xlabel("false positive error rate")
        ax.set_ylabel("false negative error rate")
        ax.set_title("best = {}".format(selection))
        # TODO end quarantine zone

    # Save ROC plot.
    if save is None:
        pylab.show()
    else:
        pylab.savefig(save)
        pylab.close(fig)

    return error


# TODO clean temporary zone.

def view_roc_curve_(params, save=None):
    """Plot ROC curve."""

    fprs = load_data(params, 'false-positive-rates')
    tprs = load_data(params, 'true-positive-rates')

    fpers = load_data(params, 'false-positive-error-rates')
    fners = load_data(params, 'false-negative-error-rates')
    fpers = 100.0 * fpers
    fners = 100.0 * fners

    # TODO clean temporary zone.
    # res = None
    # error = None
    sc_fpers = load_data(params, 'sc-false-positive-error-rates')
    sc_fners = load_data(params, 'sc-false-negative-error-rates')
    sc_fper = load_data(params, 'sc-best-false-positive-error-rate')
    sc_fner = load_data(params, 'sc-best-false-negative-error-rate')
    selection = load_data(params, 'selection')
    sc_fpers = 100.0 * sc_fpers
    sc_fners = 100.0 * sc_fners
    sc_fper = 100.0 * sc_fper
    sc_fner = 100.0 * sc_fner
    # TODO end temporary zone.

    anot_size = 8

    fig = pylab.figure()
    pylab.subplots_adjust(wspace=0.3)

    ax = fig.add_subplot(121)
    ax.plot([0.0, 1.0], [0.0, 1.0], color='black', linestyle='dashed')
    ax.plot(fprs, tprs, color='blue', linestyle='solid', zorder=3)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_title("ROC curve")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")

    ax = fig.add_subplot(122)
    # # Plot the performances of the BEER.
    ax.plot(fpers, fners)
    # # Plot the performances of each templates.
    # ax.scatter(res[:, 1], res[:, 0])
    ax.scatter(sc_fpers, sc_fners)
    # for i in range(res.shape[0]):
    for i in range(len(sc_fpers)):
        txt = str(i)
        # pos = (res[i, 1], res[i, 0])
        pos = (sc_fpers[i], sc_fners[i])
        ax.annotate(txt, pos, horizontalalignment=True, verticalalignment=True, size=anot_size)
    # # Plot the performances of the best aggregations of templates.
    # ax.scatter(error[1], error[0])
    ax.scatter(sc_fper, sc_fner)
    # pos = (error[1], error[0])
    pos = (sc_fper, sc_fner)
    ax.annotate("best", pos, horizontalalignment=True, verticalalignment=True, size=anot_size)
    # # Enhance figure.
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.grid(True)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel("false positive error rate")
    ax.set_ylabel("false negative error rate")
    ax.set_title("best = {}".format(selection))

    # Save ROC plot.
    if save is None:
        pylab.show()
    else:
        pylab.savefig(save)
        pylab.close(fig)

    # return error
    return numpy.array([sc_fner, sc_fper])

# TODO end temporary zone.
