import fastf1 as ff1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import fastf1.plotting
from fastf1.utils import delta_time
from new_utils import get_track_status_by_lap, get_weather_data_by_lap

def plot_track_status_highlights(ax, session):
    """
    Adds colored background highlights to a plot based on track status (SC/VSC).

    Args:
        ax: The matplotlib axes object to add the highlights to.
        session: The fastf1 session object.
    """
    track_status_df = get_track_status_by_lap(session)
    sc_laps = track_status_df[track_status_df['TrackStatus'].isin(['4'])]['LapNumber']
    vsc_laps = track_status_df[track_status_df['TrackStatus'].isin(['6', '7'])]['LapNumber']

    for lap in sc_laps:
        ax.axvspan(lap - 0.5, lap + 0.5, color='yellow', alpha=0.3)

    for lap in vsc_laps:
        ax.axvspan(lap - 0.5, lap + 0.5, color='orange', alpha=0.3)

def plot_rainfall_highlights(ax, session, rainfall_threshold=0.1):
    """
    Adds a blue background highlight to a plot for laps with rainfall above a threshold.

    Args:
        ax: The matplotlib axes object to add the highlights to.
        session: The fastf1 session object.
        rainfall_threshold: The rainfall value (in mm) above which to highlight laps.
    """
    weather_df = get_weather_data_by_lap(session)
    rainy_laps = weather_df[weather_df['Rainfall'] >= rainfall_threshold]['LapNumber']

    for lap in rainy_laps:
        ax.axvspan(lap - 0.5, lap + 0.5, color='blue', alpha=0.2)

def plot_race_trace(session, relative_to='average', drivers_to_plot=None):
    """
    Plots the race progression for each driver relative to a reference.

    Args:
        session: The fastf1 session object.
        relative_to: A string that can be a driver's code (e.g., 'VER'),
                     'leader', or 'average' (default).
    """
    session.load()
    laps = session.laps.copy()
    
    # Convert object-dtype columns with boolean values to bool
    for col in laps.columns:
        if laps[col].dtype == 'object' and all(isinstance(x, bool) or pd.isna(x) for x in laps[col].unique()):
            laps[col] = laps[col].astype(bool)

    # Calculate cumulative time for each driver
    laps['CumulativeTime'] = laps.groupby('Driver')['LapTime'].cumsum()

    if relative_to == 'average':
        # Calculate the average cumulative time for each lap
        reference_times = laps.groupby('LapNumber')['CumulativeTime'].mean()
        title = "Race Progression Relative to Average"
        ylabel = "Time Delta to Average (s)"
    elif relative_to == 'leader':
        # Find the leader (minimum cumulative time) at each lap
        leader_laps = laps.loc[laps.groupby('LapNumber')['CumulativeTime'].idxmin()]
        reference_times = leader_laps.set_index('LapNumber')['CumulativeTime']
        title = "Race Gaps to Leader"
        ylabel = "Gap to Leader (s)"
    else:
        # Use a specific driver as the reference
        driver_laps = laps[laps['Driver'] == relative_to]
        if driver_laps.empty:
            print(f"Driver {relative_to} not found.")
            return
        reference_times = driver_laps.set_index('LapNumber')['CumulativeTime']
        title = f"Race Gaps Relative to {relative_to}"
        ylabel = f"Gap to {relative_to} (s)"

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_title(title)
    ax.set_xlabel("Lap Number")
    ax.set_ylabel(ylabel)

    plot_track_status_highlights(ax, session)

    driver_abbreviations = drivers_to_plot if drivers_to_plot else laps['Driver'].unique()

    for driver in driver_abbreviations:
        driver_laps = laps[laps['Driver'] == driver]

        if not driver_laps.empty:
            # Use a merge to align driver laps with the reference time for each lap
            gaps_df = pd.merge(driver_laps[['LapNumber', 'CumulativeTime']],
                               reference_times.rename('ReferenceTime'),
                               on='LapNumber',
                               how='inner')

            # Calculate the gap in seconds
            gaps_df['Gap'] = (gaps_df['CumulativeTime'] - gaps_df['ReferenceTime']).dt.total_seconds()

            driver_style = fastf1.plotting.get_driver_style(driver, style=['color', 'linestyle'], session=session)
            ax.plot(gaps_df['LapNumber'], gaps_df['Gap'], label=driver, **driver_style)

    ax.legend()
    ax.grid(True)

    # Invert y-axis to show the leader/fastest driver at the top.
    ax.invert_yaxis()

    return fig, ax


def plot_telemetry_comparison(session, lap_1_driver, lap_2_driver, lap_1_number, lap_2_number, plots_to_show=['Speed', 'TimeDelta']):
    """
    Compares the telemetry of two specific laps.

    Args:
        session: The fastf1 session object.
        lap_1_driver: The driver abbreviation for the first lap (e.g., 'VER').
        lap_2_driver: The driver abbreviation for the second lap (e.g., 'HAM').
        lap_1_number: The lap number for the first lap.
        lap_2_number: The lap number for the second lap.
        plots_to_show: A list of telemetry traces to display. 
                       Options: ['Speed', 'Throttle', 'Brake', 'TimeDelta'].
    
    Returns:
        The matplotlib figure and axes objects.
    """
    session.load()
    
    lap1 = session.laps.pick_drivers(lap_1_driver).pick_lap(lap_1_number)
    lap2 = session.laps.pick_drivers(lap_2_driver).pick_lap(lap_2_number)

    tel1 = lap1.get_telemetry()
    tel2 = lap2.get_telemetry()

    # Calculate time delta
    delta_t, ref_tel, cmp_tel = delta_time(lap1, lap2)

    fig, axes = plt.subplots(len(plots_to_show), 1, figsize=(15, 5 * len(plots_to_show)), sharex=True)
    
    fig.suptitle(f"Telemetry Comparison: {lap_1_driver} Lap {lap_1_number} vs. {lap_2_driver} Lap {lap_2_number}", size=16)

    plot_map = {
        'Speed': {'y_label': 'Speed (km/h)', 'data1': tel1['Speed'], 'data2': tel2['Speed']},
        'Throttle': {'y_label': 'Throttle (%)', 'data1': tel1['Throttle'], 'data2': tel2['Throttle']},
        'Brake': {'y_label': 'Brake', 'data1': tel1['Brake'], 'data2': tel2['Brake']},
        'TimeDelta': {'y_label': f'Time Delta ({lap_2_driver} to {lap_1_driver})', 'data1': delta_t, 'data2': None}
    }

    driver_style_1 = fastf1.plotting.get_driver_style(lap1['Driver'], style=['color', 'linestyle'], session=session)
    driver_style_2 = fastf1.plotting.get_driver_style(lap2['Driver'], style=['color', 'linestyle'], session=session)

    for i, plot_type in enumerate(plots_to_show):
        ax = axes[i] if len(plots_to_show) > 1 else axes
        plot_info = plot_map[plot_type]

        if plot_type == 'TimeDelta':
            ax.plot(ref_tel['Distance'], plot_info['data1'], label=f'Time Delta ({lap_2_driver} to {lap_1_driver})', color='white')
            ax.axhline(0, color='white', linestyle='--', linewidth=1)
        else:
            ax.plot(tel1['Distance'], plot_info['data1'], label=f'{lap_1_driver} Lap {lap_1_number}', **driver_style_1)
            if plot_info['data2'] is not None:
                ax.plot(tel2['Distance'], plot_info['data2'], label=f'{lap_2_driver} Lap {lap_2_number}', **driver_style_2)
        
        ax.set_ylabel(plot_info['y_label'])
        ax.legend()
        ax.grid(True)

    # Set common x-label
    if len(plots_to_show) > 1:
        axes[-1].set_xlabel('Distance (m)')
    else:
        axes.set_xlabel('Distance (m)')

    return fig, axes


def plot_tyre_strategy(session, drivers_to_plot=None):
    """
    Plots the tyre strategy for each driver in the session.

    Args:
        session: The fastf1 session object.
        drivers_to_plot: A list of driver abbreviations to plot.
                         If None, all drivers are plotted.
    """
    session.load()
    laps = session.laps

    if drivers_to_plot is None:
        drivers_to_plot = [session.get_driver(driver_id)['Abbreviation'] for driver_id in session.drivers]

    stints = laps[['Driver', 'Stint', 'Compound', 'LapNumber']]
    stints = stints.groupby(['Driver', 'Stint', 'Compound'])['LapNumber'].agg(['min', 'max']).reset_index()
    stints = stints[stints['Driver'].isin(drivers_to_plot)]

    fig, ax = plt.subplots(figsize=(15, 10))

    for driver in drivers_to_plot:
        driver_stints = stints[stints['Driver'] == driver]
        for _, stint in driver_stints.iterrows():
            start = stint['min']
            end = stint['max']
            compound = stint['Compound']
            
            ax.barh(
                y=driver,
                width=end - start + 1,
                left=start,
                color=ff1.plotting.COMPOUND_COLORS.get(compound, '#FFFFFF'),
                edgecolor="black",
                fill=True,
                height=0.6
            )
            ax.text(
                x=start + (end - start + 1) / 2,
                y=driver,
                s=compound,
                ha='center',
                va='center',
                color='black',
                fontsize=10,
                fontweight='bold'
            )

    ax.set_title(f'{session.event.year} {session.event.EventName} - Tyre Strategy')
    ax.set_xlabel("Lap Number")
    ax.set_ylabel("Driver")
    ax.grid(False)
    ax.invert_yaxis()

    # Create custom legend
    legend_handles = [
        plt.Rectangle((0,0),1,1, color=color, edgecolor='black')
        for compound, color in ff1.plotting.COMPOUND_COLORS.items()
    ]
    ax.legend(legend_handles, ff1.plotting.COMPOUND_COLORS.keys(), title="Tyre Compounds", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    return fig, ax

def plot_lap_times(session, drivers_to_plot=None, ignore_pit_laps=True, ignore_safety_car_laps=True, ignore_first_lap=True):
    """
    Plots the lap times for each driver in the session.

    Args:
        session: The fastf1 session object.
        drivers_to_plot: A list of driver abbreviations to plot.
                         If None, all drivers are plotted.
        ignore_pit_laps: If True, laps where the driver pitted in or out
                         will be excluded from the plot.
        ignore_safety_car_laps: If True, laps that occurred under a 
                                Safety Car or Virtual Safety Car period 
                                will be excluded from the plot.
        ignore_first_lap: If True, the first lap will be excluded.
    """
    session.load()
    laps = session.laps

    if drivers_to_plot is None:
        drivers_to_plot = [session.get_driver(driver_id)['Abbreviation'] for driver_id in session.drivers]

    # if ignore_safety_car_laps:
        # track_status_df = get_track_status_by_lap(session)
        # sc_vsc_laps = track_status_df[track_status_df['TrackStatus'].isin(['4', '6', '7'])]['LapNumber']
        # laps = laps[~laps['LapNumber'].isin(sc_vsc_laps)]

    fig, ax = plt.subplots(figsize=(15, 10))

    for driver in drivers_to_plot:
        driver_laps = laps.pick_drivers(driver)
        if ignore_first_lap:
            driver_laps = driver_laps[driver_laps['LapNumber'] > 1]
        if ignore_safety_car_laps:
            driver_laps = driver_laps[~driver_laps['TrackStatus'].isin(['4', '6', '7'])]
        if ignore_pit_laps:
            driver_laps = driver_laps.loc[driver_laps['PitInTime'].isnull() & driver_laps['PitOutTime'].isnull()]
        driver_style = fastf1.plotting.get_driver_style(driver, style=['color', 'linestyle'], session=session)
        ax.plot(driver_laps['LapNumber'], driver_laps['LapTime'].dt.total_seconds(), label=driver, **driver_style)

    ax.set_title(f'{session.event.year} {session.event.EventName} - Lap Times')
    ax.set_xlabel("Lap Number")
    ax.set_ylabel("Lap Time (s)")
    ax.legend()
    ax.grid(True)

    return fig, ax


if __name__ == '__main__':
    # Enable the cache
    ff1.Cache.enable_cache('~/fastf1_cache')

    # Load a session
    session = ff1.get_session(2025, 'Great Britain', 'R')

    # Plot with default behavior (relative to average)
    # fig, ax = plot_race_trace(session)
    # plot_track_status_highlights(ax, session)
    # plt.show()

    # Plot relative to the leader
    # fig, ax = plot_race_trace(session, relative_to='leader')
    # plt.show()

    # Plot relative to a specific driver (e.g., 'VER')
    # fig, ax = plot_race_trace(session, relative_to='VER')
    # plt.show()

    # Plot with a custom list of drivers
    # fig, ax = plot_race_trace(session, drivers_to_plot=['VER', 'HAM', 'LEC'])
    # plt.show()

    # Compare telemetry of two laps
    # fig, ax = plot_telemetry_comparison(session, 'LEC', 'HAM', 10, 12)
    # plt.show()

    # Compare telemetry with custom plots
    # fig, ax = plot_telemetry_comparison(session, 'LEC', 'HAM', 10, 12, plots_to_show=['Speed', 'Throttle'])
    # plt.show()

    # Plot tyre strategy
    # fig, ax = plot_tyre_strategy(session)
    # plt.show()

    # Plot lap times
    fig, ax = plot_lap_times(session)
    plt.show()
