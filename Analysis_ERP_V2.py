# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:23:46 2026

@author: tvanneau
"""

import matplotlib as mpl

new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

# Loading needed toolboxes
import mne
import numpy as np
import pandas as pd
import os
from collections import defaultdict
import matplotlib.pyplot as plt

# ----------------------------
# File loading
# ----------------------------
list_file_TD = os.listdir('Z://Analysis/M-PRO/Epochs/TD')
file_path_TD = 'Z://Analysis/M-PRO/Epochs/TD'
list_files_epochs_TD = [s for s in list_file_TD if '_epo' in s]

list_file_ASD = os.listdir('Z://Analysis/M-PRO/Epochs/ASD')
file_path_ASD = 'Z://Analysis/M-PRO/Epochs/ASD'
list_files_epochs_ASD = [s for s in list_file_ASD if '_epo' in s]

# Load epochs files for all subjects
epochs_lst_TD = []
for f in list_files_epochs_TD:
    f_name = os.path.join(file_path_TD, f)
    epoch = mne.read_epochs(f_name)
    epoch.set_eeg_reference(ref_channels='average') # For common average reference
  #  epoch.set_eeg_reference(ref_channels=[''])

    epoch.apply_baseline((-0.05, 0.0)) # Determine the time-window for baseline-correction
    epochs_lst_TD.append(epoch)
    
# Load epochs files for all subjects
epochs_lst_ASD = []
for f in list_files_epochs_ASD:
    f_name = os.path.join(file_path_ASD, f)
    epoch = mne.read_epochs(f_name)
    epoch.set_eeg_reference(ref_channels='average')
   # epoch.set_eeg_reference(ref_channels=['PO8'])

    epoch.apply_baseline((-0.05, 0.0))
    epochs_lst_ASD.append(epoch)
    
# Triggers information
event_id_remapped = {
    'Std/4pct': 10,
    'Std/8pct': 20,
    'Std/16pct': 30,
    'deviant_ISI/4pct':    110,
    'deviant_ISI/8pct':    120,
    'deviant_ISI/16pct':   130,
    'deviant_pitch/4pct':  210,
    'deviant_pitch/8pct':  220,
    'deviant_pitch/16pct': 230,
}

#%% ERP 

# =========================================================
# Calculate participant ERPs and grand averages
# =========================================================

conditions = [
    'Std/4pct',
    'Std/8pct',
    'Std/16pct',
    'deviant_ISI/4pct',
    'deviant_ISI/8pct',
    'deviant_ISI/16pct',
    'deviant_pitch/4pct',
    'deviant_pitch/8pct',
    'deviant_pitch/16pct',
]


def compute_grand_averages(epochs_list, conditions):
    """
    Calculate one ERP per participant and then the grand average
    for each experimental condition.
    """
    participant_evokeds = defaultdict(list)

    for subject_number, epochs in enumerate(epochs_list):

        for condition in conditions:

            if condition in epochs.event_id:
                evoked = epochs[condition].average()
                participant_evokeds[condition].append(evoked)
            else:
                print(
                    f'Warning: subject {subject_number + 1} '
                    f'does not contain condition {condition}'
                )

    grand_averages = {}

    for condition, evoked_list in participant_evokeds.items():

        if len(evoked_list) > 0:
            grand_averages[condition] = mne.grand_average(
                evoked_list,
                interpolate_bads=True,
                drop_bads=True
            )

            print(
                f'{condition}: grand average based on '
                f'{len(evoked_list)} participants'
            )

    return participant_evokeds, grand_averages


# TD group
participant_evokeds_TD, grand_averages_TD = compute_grand_averages(
    epochs_lst_TD,
    conditions
)

# ASD group
participant_evokeds_ASD, grand_averages_ASD = compute_grand_averages(
    epochs_lst_ASD,
    conditions
)

#%% Topographical representation of each conditions

# =========================================================
# Topographical plots of group grand averages
# =========================================================

# Time windows in seconds
topography_windows = {
    '50–100 ms':  (0.050, 0.100),
    '100–150 ms': (0.100, 0.150),
    '150–250 ms': (0.150, 0.250),
    '250–400 ms': (0.250, 0.400),
}


def mean_topography(evoked, tmin, tmax):
    """
    Average EEG amplitude over a time window.
    Returns EEG data in µV and the corresponding EEG Info object.
    """
    evoked_eeg = evoked.copy().pick('eeg')

    time_mask = (
        (evoked_eeg.times >= tmin) &
        (evoked_eeg.times <= tmax)
    )

    if not np.any(time_mask):
        raise ValueError(
            f'No samples found between {tmin} and {tmax} seconds.'
        )

    data = evoked_eeg.data[:, time_mask].mean(axis=1) * 1e6

    return data, evoked_eeg.info

def plot_condition_topographies(
    condition,
    condition_label,
    grand_averages_TD,
    grand_averages_ASD,
    windows,
    vlim=None
):
    """
    Plot TD and ASD topographies for one condition.

    Parameters
    ----------
    condition : str
        Event name, e.g. 'Std/4pct'.
    condition_label : str
        Label displayed in the figure.
    vlim : tuple or None
        Color limits in µV, e.g. (-3, 3).
        If None, symmetric limits are calculated from both groups.
    """

    group_evokeds = {
        'TD': grand_averages_TD[condition],
        'ASD': grand_averages_ASD[condition],
    }

    topographic_data = {}

    # Calculate all time-window averages
    for group_name, evoked in group_evokeds.items():

        topographic_data[group_name] = {}

        for window_label, (tmin, tmax) in windows.items():
            data, info = mean_topography(evoked, tmin, tmax)

            topographic_data[group_name][window_label] = {
                'data': data,
                'info': info,
            }

    # Use a common symmetric scale across groups and time windows
    if vlim is None:
        all_values = np.concatenate([
            result['data']
            for group_results in topographic_data.values()
            for result in group_results.values()
        ])

        vmax = np.max(np.abs(all_values))
        vlim = (-0.7, 0.7)

    n_windows = len(windows)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=n_windows,
        figsize=(3.2 * n_windows, 6)
    )

    # Make axes consistently two-dimensional
    axes = np.atleast_2d(axes)

    im = None

    for row, group_name in enumerate(['TD', 'ASD']):

        for col, window_label in enumerate(windows):

            result = topographic_data[group_name][window_label]

            im, _ = mne.viz.plot_topomap(
                result['data'],
                result['info'],
                axes=axes[row, col],
                show=False,
                contours=6,
                cmap='RdBu_r',
                vlim=vlim,
                sensors=True
            )

            if row == 0:
                axes[row, col].set_title(
                    window_label,
                    fontsize=11
                )

            if col == 0:
                axes[row, col].set_ylabel(
                    group_name,
                    fontsize=12,
                    fontweight='bold'
                )

    # Common colorbar for the entire figure
    colorbar = fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        shrink=0.75,
        pad=0.03
    )
    colorbar.set_label('Amplitude (µV)', fontsize=11)

    fig.suptitle(
        condition_label,
        fontsize=14,
        fontweight='bold'
    )

    plt.show()
    
condition_labels = {
    'Std/4pct':              'Standard — 4%',
    'Std/8pct':              'Standard — 8%',
    'Std/16pct':             'Standard — 16%',
    'deviant_ISI/4pct':      'ISI deviant — 4%',
    'deviant_ISI/8pct':      'ISI deviant — 8%',
    'deviant_ISI/16pct':     'ISI deviant — 16%',
    'deviant_pitch/4pct':    'Pitch deviant — 4%',
    'deviant_pitch/8pct':    'Pitch deviant — 8%',
    'deviant_pitch/16pct':   'Pitch deviant — 16%',
}

for condition, label in condition_labels.items():

    if (
        condition in grand_averages_TD
        and condition in grand_averages_ASD
    ):
        plot_condition_topographies(
            condition=condition,
            condition_label=label,
            grand_averages_TD=grand_averages_TD,
            grand_averages_ASD=grand_averages_ASD,
            windows=topography_windows
        )

#%%

# =========================================================
# Plot grand-average ERPs
# =========================================================

channels = ['D4', 'D2', 'D14', 'C26'] # Need to check with the preprint paper which electrodes they are using
#channels = ['Fz', 'FCz', 'FC1', 'FC2'] 
def average_channels(evoked, ch_names):
    """Average selected EEG channels and convert V to µV."""

    missing_channels = [
        channel for channel in ch_names
        if channel not in evoked.ch_names
    ]

    if missing_channels:
        raise ValueError(
            f'Channels missing from the Evoked object: {missing_channels}'
        )

    indices = mne.pick_channels(
        evoked.ch_names,
        include=ch_names,
        ordered=True
    )

    return evoked.data[indices].mean(axis=0) * 1e6


plot_conditions = {
    'Standard 4%':   'Std/4pct',
    'Standard 8%':   'Std/8pct',
    'Standard 16%':  'Std/16pct',
#    'ISI 4%':        'deviant_ISI/4pct',
 #   'ISI 8%':        'deviant_ISI/8pct',
  #  'ISI 16%':       'deviant_ISI/16pct',
    'Pitch 4%':      'deviant_pitch/4pct',
    'Pitch 8%':      'deviant_pitch/8pct',
    'Pitch 16%':     'deviant_pitch/16pct',
}

colors = {
    'Standard 4%':  'lightskyblue',
    'Standard 8%':  'royalblue',
    'Standard 16%': 'navy',

#    'ISI 4%':       'lightgreen',
 #   'ISI 8%':       'mediumseagreen',
  #  'ISI 16%':      'darkgreen',

    'Pitch 4%':     'salmon',
    'Pitch 8%':     'tomato',
    'Pitch 16%':    'darkred',
}

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(15, 5),
    sharex=True,
    sharey=True
)

groups = {
    'TD': grand_averages_TD,
    'ASD': grand_averages_ASD,
}

for ax, (group_name, grand_averages) in zip(axes, groups.items()):

    for label, condition in plot_conditions.items():

        if condition not in grand_averages:
            continue

        evoked = grand_averages[condition]
        times = evoked.times * 1000
        data = average_channels(evoked, channels)
        std_dev = np.std(data)
        if group_name == 'TD':
            n = 15
        if group_name == 'ASD':
            n = 19
        std_error = std_dev / np.sqrt(n)
        ax.fill_between(times, data-std_error, data+std_error,
            alpha = 0.5,
            color=colors[label]
                         
                         )

        ax.plot(
            times,
            data,
            label=label,
            color=colors[label],
            linewidth=2
            
        
        )
       
        
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(0, color='black', linewidth=0.8)
   
    

    ax.set_xlim(-100, 500)
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.axvspan(100, 200, color='yellow', alpha=0.2)
    ax.set_title(
        f'{group_name} grand average',
        fontsize=13,
        fontweight='bold'
    
    )
    

axes[0].set_ylabel('Amplitude (µV)', fontsize=12)

axes[1].legend(
    ncol=3,
    fontsize=9,
    frameon=True,
    bbox_to_anchor=(1.02, 1),
    loc='upper left'
)

fig.suptitle(
    f'ERP — average of {", ".join(channels)}',
    fontsize=14,
    fontweight='bold'
)

#[this_ax.axvspan(0,100,data,color="b",alpha=0.1) for this_ax in axes]
plt.tight_layout()
plt.show()
#%%
#yes i am doing this in least efficiant way possible. i didnt sleep well
dev_4 = grand_averages_ASD.get('deviant_pitch/4pct') 
dev_4 = average_channels(dev_4, channels)
dev_8 = grand_averages_ASD.get('deviant_pitch/8pct') 
dev_8 = average_channels(dev_8, channels)
dev_16 = grand_averages_ASD.get('deviant_pitch/16pct') 
dev_16 = average_channels(dev_16, channels)
std_4 = grand_averages_ASD.get('Std/4pct')
std_4 = average_channels(std_4, channels)
std_8 = grand_averages_ASD.get('Std/8pct')
std_8 = average_channels(std_8, channels)
std_16 = grand_averages_ASD.get('Std/16pct')
std_16 = average_channels(std_16, channels)
dif_4 = dev_4 - std_4
dif_8 = dev_8 - std_8
dif_16 = dev_16 - std_16

plt.plot(times, dif_4, label='dif 4', color = 'blue')
plt.plot(times, dif_8, label = 'dif 8', color = 'r')
plt.plot(times, dif_16, label = 'dif 16', color = 'y')
plt.title('ASD Difference Wave')
plt.xlim(-100, 500)
plt.ylim(-1.5,.75)
plt.xlabel('Time (ms)', fontsize=12)
plt.ylabel('Amplitude')
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.legend()
plt.tight_layout()
plt.show()
#%%
dev_4 = grand_averages_TD.get('deviant_pitch/4pct') 
dev_4 = average_channels(dev_4, channels)
dev_8 = grand_averages_TD.get('deviant_pitch/8pct') 
dev_8 = average_channels(dev_8, channels)
dev_16 = grand_averages_TD.get('deviant_pitch/16pct') 
dev_16 = average_channels(dev_16, channels)
std_4 = grand_averages_TD.get('Std/4pct')
std_4 = average_channels(std_4, channels)
std_8 = grand_averages_TD.get('Std/8pct')
std_8 = average_channels(std_8, channels)
std_16 = grand_averages_TD.get('Std/16pct')
std_16 = average_channels(std_16, channels)
dif_4 = dev_4 - std_4
dif_8 = dev_8 - std_8
dif_16 = dev_16 - std_16
plt.figure()
plt.plot(times, dif_4, label='TD dif 4', color = 'blue')
plt.plot(times, dif_8, label = 'TD dif 8', color = 'r')
plt.plot(times, dif_16, label = 'TD dif 16', color = 'y')
plt.title('TD Difference Wave')
plt.xlim(-100, 500)
plt.xlabel('Time (ms)', fontsize=12)
plt.ylabel('Amplitude')
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.legend()
plt.ylim(-1.5,.75)
plt.tight_layout()
plt.show()
#%%
difference_conditions = {
    '4% Difference': {
        'deviant': 'deviant_pitch/4pct',
        'standard': 'Std/4pct'
    },
    '8% Difference': {
        'deviant': 'deviant_pitch/8pct',
        'standard': 'Std/8pct'
    },
    '16% Difference': {
        'deviant': 'deviant_pitch/16pct',
        'standard': 'Std/16pct'
    }
}


difference_colors = {
    '4% Difference': 'salmon',
    '8% Difference': 'tomato',
    '16% Difference': 'darkred'
}


difference_waves = {
    'TD': {},
    'ASD': {}
}


for group_name, grand_averages in groups.items():

    print(f'\nCreating difference waves for {group_name}')

    for label, condition_names in difference_conditions.items():

        deviant_condition = condition_names['deviant']
        standard_condition = condition_names['standard']

        missing_conditions = [
            condition
            for condition in [
                deviant_condition,
                standard_condition
            ]
            if condition not in grand_averages
        ]

        if missing_conditions:
            print(
                f'Skipping {group_name} {label}. '
                f'Missing conditions: {missing_conditions}'
            )
            continue

        deviant_evoked = grand_averages[
            deviant_condition
        ]

        standard_evoked = grand_averages[
            standard_condition
        ]

        # Confirm that the time vectors match before subtraction
        if not np.allclose(
            deviant_evoked.times,
            standard_evoked.times
        ):
            raise ValueError(
                f'Time vectors do not match for '
                f'{group_name} {label}.'
            )

        # Confirm that channel order matches before subtraction
        if deviant_evoked.ch_names != standard_evoked.ch_names:
            raise ValueError(
                f'Channel order does not match for '
                f'{group_name} {label}.'
            )

        # Difference = deviant minus standard
        difference_evoked = mne.combine_evoked(
            [
                deviant_evoked,
                standard_evoked
            ],
            weights=[1, -1]
        )

        difference_evoked.comment = (
            f'{deviant_condition} minus '
            f'{standard_condition}'
        )

        difference_waves[group_name][label] = (
            difference_evoked
        )

        print(
            f'Created {label}: '
            f'{deviant_condition} - '
            f'{standard_condition}'
        )


# ============================================================
# PLOT DIFFERENCE WAVES
# ============================================================

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(13, 5),
    sharex=True,
    sharey=True
)


for ax, group_name in zip(
    axes,
    ['TD', 'ASD']
):

    group_differences = difference_waves[
        group_name
    ]

    for label, difference_evoked in (
        group_differences.items()
    ):

        times = difference_evoked.times * 1000

        difference_data = average_channels(
            difference_evoked,
            channels
        )

        ax.plot(
            times,
            difference_data,
            label=label,
            color=difference_colors[label],
            linewidth=2
        )

    ax.axhline(
        0,
        color='black',
        linewidth=0.8
    )

    ax.axvline(
        0,
        color='black',
        linewidth=0.8
    )

    ax.set_xlim(-100, 500)

    ax.set_xlabel(
        'Time (ms)',
        fontsize=12
    )

    ax.set_title(
        f'{group_name}: deviant − standard',
        fontsize=13,
        fontweight='bold'
    )


axes[0].set_ylabel(
    'Difference amplitude (µV)',
    fontsize=12
)


axes[1].legend(
    fontsize=10,
    frameon=True,
    bbox_to_anchor=(1.02, 1),
    loc='upper left'
)


fig.suptitle(
    (
        'Pitch difference waves '
        f'— average of {", ".join(channels)}'
    ),
    fontsize=14,
    fontweight='bold'
)


plt.tight_layout()
plt.show()