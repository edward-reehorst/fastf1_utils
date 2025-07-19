import fastf1 as ff1
import pandas as pd

def get_track_status_by_lap(session):
    """
    Gets the track status for each lap of a session, based on the leader's perspective.

    Track Status Codes:
        1: Green Flag
        2: Yellow Flag
        4: Safety Car
        5: Red Flag
        6: Virtual Safety Car Deployed
        7: Virtual Safety Car Ending

    Args:
        session: The fastf1 session object.

    Returns:
        A pandas DataFrame with lap number and the track status at that lap.
    """
    session.load()
    laps = session.laps.copy()
    laps['CumulativeTime'] = laps.groupby('Driver')['LapTime'].cumsum()
    leader_laps = laps.loc[laps.groupby('LapNumber')['CumulativeTime'].idxmin()]
    track_status = leader_laps.loc[:, ["LapNumber", "TrackStatus"]]
    return track_status

def get_weather_data_by_lap(session):
    """
    Gets the weather data for each lap of a session, based on the leader's perspective.

    Args:
        session: The fastf1 session object.

    Returns:
        A pandas DataFrame with lap number and corresponding weather data.
    """
    session.load()
    laps = session.laps.copy()
    weather_data = session.weather_data

    # Find leader laps
    laps['CumulativeTime'] = laps.groupby('Driver')['LapTime'].cumsum()
    leader_laps = laps.loc[laps.groupby('LapNumber')['CumulativeTime'].idxmin()]

    # Sort by time for merge_asof
    leader_laps = leader_laps.sort_values(by='Time')
    weather_data = weather_data.sort_values(by='Time')

    # Merge weather data based on the closest time
    weather_laps = pd.merge_asof(
        leader_laps[['LapNumber', 'Time']],
        weather_data,
        on='Time',
        direction='nearest'
    )

    # Select relevant columns
    weather_laps = weather_laps[['LapNumber', 'AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp', 'WindDirection', 'WindSpeed']]

    return weather_laps


if __name__ == '__main__':
    # Enable the cache
    ff1.Cache.enable_cache('~/fastf1_cache')

    # Load a session
    session = ff1.get_session(2023, 'Monza', 'R')

    # Get track status for each lap from the leader's perspective
    print("--- Track Status per Lap (Leader's Perspective) ---")
    track_status_df = get_track_status_by_lap(session)
    print(track_status_df)
    print('\n')

    # Get weather data for each lap from the leader's perspective
    print("--- Weather Data per Lap (Leader's Perspective) ---")
    weather_df = get_weather_data_by_lap(session)
    print(weather_df)
