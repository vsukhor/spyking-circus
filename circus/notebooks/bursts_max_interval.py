# maxinterval.R --- maxinterval burst detection (from Neuroexplorer).
# Author: Stephen Eglen
# Copyright: GPL
# Fri 23 Feb 2007

from typing import Final, NamedTuple
import matplotlib.pyplot as plt

import numpy as np

from circus.notebooks.basic_analysis import extract_time_data

test_seq_t = [np.array([1, 50, 60, 120, 130, 345, 365, 530, 670, 690, 750, 790, 799], dtype=float) / 1000.,
              np.array([23, 400, 560, 580, 590, 900, 940, 1090, 1120, 1150, 1160, 2300, 2330, 2350, 2380, 3000], dtype=float) / 1000.,
              np.array([12, 234, 256, 300, 500, 610, 632, 651, 800, 1000], dtype=float) / 1000.]
from circus.notebooks.basic_analysis import spike_intervals
test_isi = spike_intervals(test_seq_t)
test_mask = [np.array([0,0,0,0,0, 1, 2, 3, 4,4,4,4,4]),
             np.array([0, 1, 2,2,2, 3, 4, 5,5,5,5, 6,6,6,6, 7]),
             np.array([0, 1,1,1, 2, 3,3,3, 4, 5])]
test_bsize = [np.array([5,5,5,5,5, 1, 1, 1, 5,5,5,5,5]),
              np.array([1, 1, 3,3,3, 1, 1, 4,4,4,4, 4,4,4,4, 1]),
              np.array([1, 3,3,3, 1, 3,3,3, 1, 1])]


class Pars(NamedTuple):
    """Parameters (ms)
    """

    max_isi_first = 100   # max ISI at burst start
    max_isi_last = 100    # max ISI at burst end
    max_isi_intnl = max(max_isi_first, max_isi_last)
    min_ibi = 0            # min inter-burst interval
    min_dur = 0            # min burst duration
    min_nspikes = 3 # min nunber of spikes in burst
    min_nisis = min_nspikes - 1

    @classmethod
    def is_beg_isi_ok(cls, isi):

        return isi < cls.max_isi_first

    @classmethod
    def is_end_isi_ok(cls, isi):

        return isi < cls.max_isi_last

    @classmethod
    def is_int_isi_ok(cls, isi):

        return np.alltrue(isi < cls.max_isi_intnl)

    @classmethod
    def is_burst(cls, isis):
        return cls.is_beg_isi_ok(isis[0]) and \
               cls.is_end_isi_ok(isis[-1]) and \
               cls.is_int_isi_ok(isis[1:-2])


class Bursts:

    beg : np.ndarray
    end : np.ndarray
    ibi : np.ndarray
    num = 0

    def __init__(self, s: int):

        self.beg = np.empty(s)
        self.end = np.empty(s)
        self.ibi = np.empty(s)

    def append(self, be, en, ib):
        self.beg[self.num] = be
        self.end[self.num] = en
        self.ibi[self.num] = ib
        self.num += 1


class Detector:

    # Create a temp array for the storage of the bursts. Assume that
    # it will not be longer than nspikes/2 since we need at least two
    # spikes to be in a burst.
    bursts : Bursts
    _burst = None    # current burst number

    def __init__(self,
                 filename: np.array,
                 pars: Pars,
                 template_id):

        _, \
        spike_times, \
        spike_intervals, \
        _ = extract_time_data(filename, template_id)

        self.burst_indx = self.find_bursts(spike_times, spike_intervals, pars)
        self.sizes = self.set_sizes(self.burst_indx)


    @classmethod
    def find_bursts(cls,
                    spike_times,
                    spike_intervals,
                    pars : Pars):

        """ For single spike train, finds the burst using max interval method.
        # params currently in par
        ##

        # TODO: all our burst analysis routines should use the same
        # value to indiciate "no bursts" found.
        # no.bursts = NA;                  #value to return if no bursts found.
        no_bursts = matrix(nrow=0,ncol=1)  #emtpy value nrow()=length() = 0.
        """

        indx = [np.full_like(t, np.nan, dtype=np.int)
                     for t in spike_times]

        for ti, si in enumerate(spike_intervals):
            ns = len(si) + 1   # number of spikes
            id = indx[ti]
            bi = 0       # burst index
            ii = 0       # starting spike index
            while ii < ns:
                # burst length (spikes):
                bl = pars.min_nspikes \
                    if ii <= ns - pars.min_nspikes \
                    else 1
                while bl <= ns - ii:
                    if ii <= ns - pars.min_nspikes:
                        isis = si[ii:ii+bl-1]
                        isburst = pars.is_burst(isis)
                    else:
                        isburst = False
                    if isburst:
                        bl += 1
                    if not isburst or bl + ii > ns:
                        if bl > pars.min_nspikes:
                            bl -= 1
                        elif bl > 1:
                            bl = 1
                        id[ii:ii+bl] = bi
                        ii += bl
                        bi += 1
                        break

        return indx

    @classmethod
    def set_sizes(cls, burst_indx):

        sizes = [np.empty(bi[-1]+1, dtype=np.int) for bi in burst_indx]

        for ix, sz in zip(burst_indx, sizes):
            for n in range(sz.shape[0]):
                sz[n] = len(np.argwhere(ix==n))

        return sizes


def main_test():

    pars = Pars()

    indxs = Detector.find_bursts(test_seq_t, test_isi, pars)
#    bursts = Detector(filename, pars, template_id=None)
    sizes = Detector.set_sizes(indxs)

    fraction_of_singles = np.array([len(np.argwhere(sz==1)) / sum(sz) for sz in sizes])
    avg_fraction_of_singles = fraction_of_singles.mean()

    print('')


def make_size_distr(filename, pars):

    bd = Detector(filename, pars, None)
    indxs = bd.burst_indx
#    bursts = Detector(filename, pars, template_id=None)
    return bd.sizes


def main(filename):

    sizes = make_size_distr(filename, Pars)

    fraction_of_singles = np.array([len(np.argwhere(sz==1)) / sum(sz) for sz in sizes])
    avg_fraction_of_singles = fraction_of_singles.mean()

    show_burst_hist(sizes,
                    None,
                    bin_size=1)
    from circus.notebooks.electrode_positions import load_cluster_data
    from circus.notebooks.electrode_positions import plot_electrode_geometry

    clusters_data = load_cluster_data(filename)
    template_el_ind = clusters_data['electrodes']
    template_colors = np.empty((len(template_el_ind),3))
#    for i in template_colors:


#    plot_electrode_geometry(template_el_ind, template_colors)

    print('')

def show_burst_hist(sizes,
                    template_id,
                    bin_size=1):
    """Displays the distribution of  burst lengths as a histogram.
       If template_id is None, all available templates are used.
    """
    size_total = np.concatenate(sizes, axis=0)

    fig = plt.figure(figsize=[12, 9])
    ax = plt.subplot(1, 1, 1)
    nb_bins = int(np.ceil(np.max(size_total) / bin_size))
    maximum_interval = float(nb_bins) * bin_size
    hist_kwargs = {
        'bins' : nb_bins,
        'range' : (0.0, maximum_interval),
        'density' : True,
        'histtype' : 'step',
        'stacked' : True,
        'fill' : False

    }
    [ax.hist(a, color=None, ls='--', lw=0.5, **hist_kwargs)
     for a in sizes]
    ax.hist(size_total, color='k', ls='-', lw=0.5, **hist_kwargs)
    ax.set_xlim(0.0, maximum_interval)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 0))
    ax.set_xlabel("rel value")
    ax.set_ylabel("probability")
    if template_id is not None:
        ax.set_title(f"burst szes (template {template_id})")
    else:
        ax.set_title(f"burst sizes (all templates, black solid line: average)")
    fig.tight_layout()
    plt.show()

########################################################################
if __name__ == '__main__':

    fname = '/Users/vs/neuro/spyking_circus/test1/20160426_patch3/patch_3.raw'

    main(fname)
    print('')
