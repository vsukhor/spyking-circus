
import numpy as np
import matplotlib.pyplot as plt
from mea_256 import total_nb_channels, radius, channel_groups


def plot_electrode_geometry(template_electrodes=None,
                            template_colors=None):

    ncols =  int(np.sqrt(total_nb_channels))
    nrows = int(np.sqrt(total_nb_channels))
    fig_width, fig_height = 11, 11
    el_positions_x = [k[0] for k in channel_groups[1]['geometry'].values()]
    el_positions_y = [k[1] for k in channel_groups[1]['geometry'].values()]
    el_minx = min(el_positions_x)
    el_maxx = max(el_positions_x)
    el_miny = min(el_positions_y)
    el_maxy = max(el_positions_y)
    geom_width = el_maxx - el_minx
    geom_height = el_maxy - el_miny
    geom_step_x = geom_width / ncols
    geom_step_y = geom_height / nrows
    geom_width = geom_step_x * (ncols + 1)
    geom_height = geom_step_y * (nrows + 1)
    ax_width, ax_height = 1 / ncols, 1 / nrows
    fig = plt.figure(figsize=(10, 10))
    axs = []
    left = np.empty(ncols*nrows, dtype=float)
    bottom = np.empty(ncols*nrows, dtype=float)
    for i in range(nrows):
        for j in range(ncols):
            elind = i * ncols + j
            left[elind] = (el_positions_x[elind] - el_minx) / geom_width # j * ax_width
            bottom[elind] = (el_positions_y[elind] - el_miny) / geom_height # i * ax_height
            a = fig.add_axes([left[elind], bottom[elind], ax_width, ax_height])
            a.set_aspect('equal')
            a.axis('off')
            lims = 100
            a.set_xlim(0, lims)
            a.set_ylim(0, lims)
            center = (lims / 2, lims / 2)
            radius = 15
            if template_electrodes is None:
                circle = plt.Circle(center, radius=radius)
                if elind in channel_groups[1]['channels']:
                    circle.set_color([0.3, 0.3, 0.3])
                else:
                    circle.set_ec([0.3, 0.3, 0.3])
                    circle.set_fc('w')
                a.add_patch(circle)
            else:
                templs = np.argwhere(template_electrodes==elind)
                nc = len(templs)
                if nc > 1:
                    shift = 2 * radius
                    for ci in range(nc):
                        angle = ci * 2*np.pi / nc
                        circle = plt.Circle((center[0]+shift*np.cos(angle),
                                             center[1]+shift*np.sin(angle)),
                                            radius=radius,
                                            color=template_colors[templs[ci]])
                        a.add_patch(circle)
                elif nc == 1:
                    circle = plt.Circle(center,
                                        radius=radius,
                                        color=template_colors[templs[0]])
                    a.add_patch(circle)
                else:
                    circle = plt.Circle(center, radius=radius)
                    if elind in channel_groups[1]['channels']:
                        circle.set_ec([0.3, 0.3, 0.3])
                    else:
                        circle.set_ec([0.8, 0.8, 0.8])
                    circle.set_fc('w')
                    a.add_patch(circle)

            axs.append(a)
            annot = a.annotate(f"{elind}", xy=(0,0), xytext=(4,4),
                               textcoords="offset points", size=7)
            annot.set_visible(True)
    plt.show()

    print("")


def load_cluster_data(fname_raw):

    import h5py
    import os
    import re

    from basic_analysis import load_parameters

    params = load_parameters(fname_raw)

    sampling_rate = params.rate
    file_out_suff = params.get('data', 'file_out_suff')
    nb_channels = params.getint('data', 'N_e')
    nb_time_steps = params.getint('detection', 'N_t')

    # Load clusters.
    clusters_path = "{}.clusters.hdf5".format(file_out_suff)
    if not os.path.isfile(clusters_path):
        raise FileNotFoundError(clusters_path)
    with h5py.File(clusters_path, mode='r', libver='earliest') as f:
        clusters_data = dict()
        p = re.compile('_\d*$')  # noqa
        for key in list(f.keys()):
            m = p.search(key)
            if m is None:
                clusters_data[key] = f[key][:]
            else:
                k_start, k_stop = m.span()
                key_ = key[0:k_start]
                channel_nb = int(key[k_start+1:k_stop])
                if key_ not in clusters_data:
                    clusters_data[key_] = dict()
                clusters_data[key_][channel_nb] = f[key][:]

    return clusters_data


def main():

    filename_raw = \
        '/Users/vs/neuro/spyking_circus/test1/20160426_patch3/patch_3.raw'

    clusters_data = load_cluster_data(filename_raw)
    template_el_ind = clusters_data['electrodes']
    template_colors = np.full((len(template_el_ind),3), [0.,0.,0.])

    plot_electrode_geometry(template_el_ind, template_colors) #


########################################################################
if __name__ == '__main__':

    main()

    print('')
