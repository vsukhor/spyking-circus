"""Extracts and displays some basic spike-derived metrics.
"""

import matplotlib.pyplot as plt
import numpy as np

from circus.shared.files import load_data
from circus.shared.parser import CircusParser


def set_template_ids(template_id, results):
    """Preprocesses template ids.
    """

    template_ids = template_id if template_id is not None \
        else np.sort([int(k.split('_')[1])
                      for k in results['spiketimes'].keys()])
    if not np.iterable(template_ids):
        template_ids = [template_ids]

    return template_ids


def extract_time_data(filename,
                      template_id=None):

    # Load parameters.
    params = CircusParser(filename)
    _ = params.get_data_file()
    sampling_rate = params.rate

    # Load spike intervals.
    results = load_data(params, 'results')
    template_ids = set_template_ids(template_id, results)

    spike_times = \
        [results['spiketimes'][f'temp_{tid}'] / sampling_rate
         for tid in template_ids]

    spike_intervals = \
        [np.diff(st) * 1e+3 for st in spike_times]
    spint_total = np.concatenate(spike_intervals, axis=0)

    return template_ids, spike_times, spike_intervals, spint_total


def show_isi(filename,
             template_id,
             maximum_interval=50.0,
             bin_size=1.0):
    """Displays the distribution of inter-spike intervals.
    """

    template_ids, \
    spike_times, \
    spike_intervals, \
    spint_total = extract_time_data(filename, template_id)

    fig = plt.figure(figsize=[12, 9])
    ax = plt.subplot(1, 1, 1)
    nb_bins = int(np.ceil(maximum_interval / bin_size))
    maximum_interval = float(nb_bins) * bin_size
    hist_kwargs = {
        'bins' : nb_bins,
        'range' : (0.0, maximum_interval),
        'density' : True,
        'histtype' : 'step',
        'stacked' : True,
        'fill' : False

    }
    [ax.hist(si, color=None, ls='--', lw=0.2, **hist_kwargs)
     for si in spike_intervals]
    ax.hist(spint_total, color='k', ls='-', lw=0.5, **hist_kwargs)
    ax.set_xlim(0.0, maximum_interval)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 0))
    ax.set_xlabel("interval (ms)")
    ax.set_ylabel("probability")
    if template_id is not None:
        ax.set_title(f"ISI (template {template_id})")
    else:
        ax.set_title(f"ISI (all templates, black solid line: average)")
    fig.tight_layout()
    plt.show()


def extract_amplitude_data(filename,
                           template_id,
                           extension=None):
    """Extracts template amplitudes from the results file.
       Returns them classified by template index and accumulated.
    """

    params = CircusParser(filename)
    _ = params.get_data_file()
    extension = "" if extension is None \
        else "-" + extension
    results = load_data(params, 'results', extension=extension)
    template_ids = set_template_ids(template_id, results)

    ampl = [results['amplitudes'][f'temp_{tid}'][:, 0] for tid in template_ids]
    ampl_total = np.concatenate(ampl, axis=0)

    return ampl, ampl_total


def show_amplitudes(filename,
                    template_id,
                    extension=None):
    """Displays amplitudes vs time.
       If template_id is None, all available templates are shown.
    """

    template_ids, \
    spike_times, \
    spike_intervals, \
    spint_total = extract_time_data(filename, template_id)

    amplitudes, _ = extract_amplitude_data(filename,
                                           template_id,
                                           extension)
    fig = plt.figure(figsize=[12, 9])
    ax = plt.subplot(1, 1, 1)
    scatter_kwargs = {
        's': 1**2,
        'c': None,
    }
    [ax.scatter(st, a, **scatter_kwargs)
     for st, a in zip(spike_times, amplitudes)]
    axline_kwargs = {
        'color': 'black',
        'linewidth': 0.5,
    }
    ax.axhline(y=0.0, **axline_kwargs)
    ax.axhline(y=1.0, **axline_kwargs)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude")
    ax.set_title(f"amplitudes (template {template_id})")
    if template_id is not None:
        ax.set_title(f"amplitudes (template {template_id})")
    else:
        ax.set_title(f"amplitudes (all templates)")
    fig.tight_layout()
    plt.show()


def show_amplitude_hist(filename,
                        template_id,
                        extension=None,
                        bin_size=0.1):
    """Displays the distribution of spike amplitudes as a histogram.
       If template_id is None, all available templates are used.
    """

    ampl, ampl_total = extract_amplitude_data(filename,
                                              template_id,
                                              extension)

    fig = plt.figure(figsize=[12, 9])
    ax = plt.subplot(1, 1, 1)
    nb_bins = int(np.ceil(np.max(ampl_total) / bin_size))
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
     for a in ampl]
    ax.hist(ampl_total, color='k', ls='-', lw=0.5, **hist_kwargs)
    ax.set_xlim(0.0, maximum_interval)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 0))
    ax.set_xlabel("rel value")
    ax.set_ylabel("probability")
    if template_id is not None:
        ax.set_title(f"amplitudes (template {template_id})")
    else:
        ax.set_title(f"amplitudes (all templates, black solid line: average)")
    fig.tight_layout()
    plt.show()


########################################################################
if __name__ == '__main__':

    fname = '/Users/vs/neuro/spyking_circus/test1/20160426_patch3/patch_3.raw'

#    show_isi(fname, None)
    show_amplitudes(fname, None)
    show_amplitude_hist(fname, None)
    print('')
