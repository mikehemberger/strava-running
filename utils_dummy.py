import os
import requests
import urllib3
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.collections import PolyCollection, LineCollection
import matplotlib.colors as mcolors


# Helper function to get data from Strava
def get_strava_data(endpoint, ACCESS_TOKEN, params=None, BASE_URL="https://www.strava.com/api/v3"):
    # Base URL for Strava API
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    while True:
        response = requests.get(f"{BASE_URL}/{endpoint}", headers=headers, params=params)
        if response.status_code == 429:  # Rate limit exceeded
            print("Rate limit hit. Pausing for 15 minutes...")
            time.sleep(15 * 60)  # Wait for 15 minutes
        else:
            response.raise_for_status()
            return response.json()

# Get all activities
def get_activities(ACCESS_TOKEN):
    activities = []
    page = 1
    while True:
        params = {"per_page": 200, "page": page}
        data = get_strava_data("athlete/activities", ACCESS_TOKEN, params)
        if not data:
            break
        activities.extend(data)
        page += 1
    return activities

# Get starred segments
def get_starred_segments(ACCESS_TOKEN):
    starred_segments = []
    page = 1
    while True:
        params = {"per_page": 200, "page": page}
        data = get_strava_data("segments/starred", ACCESS_TOKEN, params)
        if not data:
            break
        starred_segments.extend(data)
        page += 1

    # Extract relevant fields
    segments = [
        {
            "name": segment["name"],
            "id": segment["id"],
            "distance": segment["distance"],# / 1000,  # ->km
            "actvity_type" : segment.get("activity_type"),
            "average_grade": segment["average_grade"],
            "maximum_grade": segment.get("maximum_grade"),  # Use .get() in case it's missing
            "total_elevation_gain": segment.get("total_elevation_gain"),
            "effort_count": segment.get("athlete_segment_stats", {}).get("effort_count", None)
        }
        for segment in starred_segments
    ]
    
    return pd.DataFrame(segments)



def collect_segment_efforts(filtered_activities, starred_segment_names, ACCESS_TOKEN):
    segment_efforts = {name: [] for name in starred_segment_names}
    for activity in filtered_activities:
        activity_id = activity["id"]
        # Introduce a delay to avoid hitting the rate limit
        time.sleep(1)  # 1-second delay
        # Get segment efforts for the activity
        activity_data = get_strava_data(f"activities/{activity_id}", ACCESS_TOKEN)
        if "segment_efforts" in activity_data:
            for effort in activity_data["segment_efforts"]:
                segment_name = effort["name"]
                if segment_name in starred_segment_names:
                    segment_efforts[segment_name].append({
                        "activity_id": activity_id,
                        "name": effort["name"],
                        "start_date_local": effort["start_date_local"],
                        "distance": effort.get("distance", None),
                        "moving_time": effort["moving_time"],
                        "start_index": effort.get("start_index", None),
                        "end_index": effort.get("end_index", None),
                        "average_cadence": effort.get("average_cadence", None),
                        "average_heartrate": effort.get("average_heartrate", None),
                    })
    return segment_efforts


def efforts_to_dataframe(segment_efforts):
    # Flatten the dictionary into a list of dictionaries
    rows = []
    for segment_name, efforts in segment_efforts.items():
        for effort in efforts:
            row = {
                "segment_name": segment_name,
                "activity_id": effort["activity_id"],
                "name": effort["name"],
                "start_date_local": effort["start_date_local"],
                "distance": effort["distance"], #/ 1000,
                "moving_time": effort["moving_time"],
                "start_index": effort["start_index"],
                "end_index": effort["end_index"],
                "average_cadence": effort["average_cadence"],
                "average_heartrate": effort["average_heartrate"],
            }
            # Add velocity and pace
            if effort["distance"] and effort["moving_time"]:
                row["velocity_kmh"] = (effort["distance"] / 1000) / (effort["moving_time"] / 3600)
                row["pace_min_per_km"] = (effort["moving_time"] / (effort["distance"] / 1000)) / 60
            else:
                row["velocity_kmh"] = None
                row["pace_min_per_km"] = None
            rows.append(row)
    
    # Create a DataFrame
    df = pd.DataFrame(rows)
    return df


def fill_with_variable_alpha(x, y1, y2, alpha_range=(0.3, 1.0), color="royalblue", ax=None):
    """Creates a fill_between effect where the transparency increases along the x-axis."""
    if ax is None:
        fig, ax = plt.subplots()
    
    alphas = np.linspace(alpha_range[0], alpha_range[1], len(x) - 1)
    ccolor = mcolors.to_rgb(color)
    
    # Create polygon vertices
    verts = [((x[i], y1[i]), (x[i], y2[i]), (x[i+1], y2[i+1]), (x[i+1], y1[i+1])) for i in range(len(x) - 1)]
    colors = [ccolor + (alphas[i],) for i in range(len(alphas))]  

    poly = PolyCollection(verts, facecolors=colors, edgecolor='none')
    ax.add_collection(poly)
    
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(min(np.nanmin(y2), np.nanmin(y1)), max(np.nanmax(y1), np.nanmax(y2)))

    return poly

def plot_with_variable_alpha(x, y, alpha_range=(0.3, 1.0), color="royalblue", ax=None, linewidth=1.5):
    """Plots a line with increasing alpha values along the x-axis."""
    if ax is None:
        fig, ax = plt.subplots()

    alphas = np.linspace(alpha_range[0], alpha_range[1], len(x) - 1)
    ccolor = mcolors.to_rgb(color)

    # Create line segments
    segments = [[(x[i], y[i]), (x[i+1], y[i+1])] for i in range(len(x) - 1)]
    colors = [ccolor + (alphas[i],) for i in range(len(alphas))]

    lc = LineCollection(segments, colors=colors, linewidths=linewidth)
    ax.add_collection(lc)
    
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(np.nanmin(y), np.nanmax(y))

    return lc


def strava_stream_to_dataframe(stream):
    """
    Convert a Strava stream dictionary to a cleaned and processed DataFrame.
    
    Parameters:
        stream (dict): The Strava stream data, where each key contains a dictionary 
                       with a "data" key holding the list of values.
    
    Returns:
        pd.DataFrame: A processed DataFrame containing the cleaned and adjusted metrics.
    """
    # Create main DataFrame from stream
    df_stream = pd.DataFrame({key: s["data"] for key, s in stream.items() if "data" in s})
    
    # Expand and merge latlng data
    if "latlng" in stream.keys():
        latlng_df = pd.DataFrame(stream["latlng"]["data"], columns=["lat", "lng"])
        df_stream = pd.concat([df_stream, latlng_df], axis=1)
        df_stream = df_stream.drop(columns=["latlng"], errors="ignore")  # Drop latlng if present

    # Adjust metrics
    df_stream["time"] = pd.to_timedelta(df_stream["time"], unit="s")  # Convert seconds to timedelta
    df_stream["time_min"] = df_stream["time"].dt.total_seconds() / 60  # time in minutes
    df_stream["distance"] = df_stream["distance"] / 1000  # Convert meters to kilometers
    df_stream["velocity_smooth"] = df_stream["velocity_smooth"] * 3.6  # Convert m/s to km/h
    
    # Clean velocity_smooth with limits
    lower_lim, upper_lim = 4, 50  # km/h
    df_stream["velocity_smooth"] = df_stream["velocity_smooth"].where(
        (df_stream["velocity_smooth"] > lower_lim) & (df_stream["velocity_smooth"] <= upper_lim), np.nan)
    
    # Calculate pace
    df_stream["pace_min"] = 60 / df_stream["velocity_smooth"]  # min/km
    df_stream["pace_sec"] = pd.to_timedelta(df_stream["pace_min"] * 60, unit="s").dt.total_seconds()  # sec/km

    return df_stream


def get_strava_activity_data(ACCESS_TOKEN, url="https://www.strava.com/api/v3/activities"):
    """ Obtain all activities from strava"""
    
    header = {'Authorization': 'Bearer ' + ACCESS_TOKEN}  # access_token = strava_tokens['access_token']
    
    # Run parameter columns
    columns = ["id", "name", "external_id",
                "start_date_local",
                "type", 'sport_type',
                "distance", "moving_time", "elapsed_time",
                "total_elevation_gain",
                'start_latlng', "end_latlng",            
                'average_speed', 'max_speed',
                'has_heartrate', 'average_heartrate', #'average_cadence',
                'suffer_score',
                #'map', 'map.summary_polyline', # these have to be indexed differently than the other params!  
    ]
    # Create the dataframe to store API-returned activity data
    activities = pd.DataFrame(columns=columns)

    # Loop across pages + loop across columns! except for map-summary_polyline
    page = 1
    pp = 200  # entries per page
    while True:
        param = {'per_page': pp, 'page': page}      # get page of activities from Strava
        r = requests.get(url, headers=header, params=param).json()
        if (not r):  # if no results then exit loop
            break
        # otherwise add new data to dataframe
        for x in range(len(r)):
            if (r[x]["has_heartrate"]):  #  & ~np.isnan(r[x]["suffer_score"])
                for col in columns:
                    activities.loc[x + (page-1)*pp, col] = r[x][col]
                    # Different indexing for map
                    # activities.loc[x + (page-1)*pp,'map'] = r[x]['map']['summary_polyline']
        page += 1  # increment page
        
    print(f"{len(activities)} activities obtained from Strava API")
    return activities



def get_strava_refresh_access_token():
    """ GET REFRESH TOKEN FOR API """
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    auth_url = "https://www.strava.com/oauth/token"

    payload = {
        'client_id': ,
        'client_secret': '',
        'refresh_token': '',
        'grant_type': "refresh_token",
        'f': 'json'
    }

    # Request token
    res = requests.post(auth_url, data=payload, verify=False)
    return res.json()['access_token']  #print("Access Token = {}\n".format(access_token))


def get_activity_by_id(activity_id, ACCESS_TOKEN):
    url = f"https://www.strava.com/api/v3/activities/{activity_id}"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching activity {activity_id}: {response.status_code}")
        print(response.text)
        return None  # use this also below but it is possible i hit the rate limit for the day


def get_activity_stream_by_id(activity_id, ACCESS_TOKEN):
    url = f"https://www.strava.com/api/v3/activities/{activity_id}/streams"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    params = {
        "keys": "time,distance,velocity_smooth,heartrate,cadence,altitude,latlng,grade_smooth",
        "key_by_type": "true"
    }
    while True:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 429:  # Rate limit exceeded
            print("Rate limit hit. Pausing for 15 minutes...")
            time.sleep(15 * 60)  # Wait for 15 minutes
        else:
            response.raise_for_status()
            return response.json()


def extract_segment_data(activity_id, segment_id, ACCESS_TOKEN):
    activity = get_activity_by_id(activity_id, ACCESS_TOKEN)
    if not activity:
        print(f"Failed to fetch activity {activity_id}")
        return None

    print(f"Segment Efforts for Activity {activity_id}: {activity['segment_efforts']}")
    idxs = list()
    for effort in activity["segment_efforts"]:
        if effort["segment"]["id"] == segment_id:
            start_idx = effort["start_index"]
            end_idx = effort["end_index"] + 1
            idxs.append((start_idx, end_idx))
            
    return idxs


def slice_stream_by_segment_indices(stream, segment_indices):

    if not stream:
        raise ValueError("Stream data is missing or invalid!")
    
    effort_dfs = []
    for idx, (start, end) in enumerate(segment_indices, start=1):
        effort_data = {key: s["data"][start:end] for key, s in stream.items() if "data" in s}
        effort_df = pd.DataFrame(effort_data)
        
        # Transform data to appropriate units
        if "time" in effort_df:
            effort_df["time"] = effort_df["time"] / 60  # seconds to minutes
            effort_df["seg_time"] = effort_df["time"] - effort_df["time"].iloc[0]
            
        if "distance" in effort_df:
            effort_df["distance"] = effort_df["distance"] #/ 1000  # meters to kilometers
            effort_df["seg_distance"] = effort_df["distance"] - effort_df["distance"].iloc[0]
        
        if "velocity_smooth" in effort_df:
            effort_df["velocity_smooth"] = effort_df["velocity_smooth"] * 3.6  # m/s to km/h
            
        if "latlng" in effort_df:
            effort_df[['latitude', 'longitude']] = effort_df['latlng'].apply(pd.Series)
            effort_df = effort_df.drop(columns=['latlng'])
        
        # Add effort label
        effort_df["effort"] = int(idx)
        effort_dfs.append(effort_df)
    
    # Combine all effort DataFrames
    df_efforts = pd.concat(effort_dfs, ignore_index=True)
    return df_efforts


def apply_pace_labels(ax, min_pace="5:00", max_pace="10:00", step="0:30", axis="y"):
    """
    Apply `min:sec` pace labeling to the specified axis.

    Parameters:
    - ax: matplotlib axis object to apply labels to.
    - min_pace: Starting pace as a string in "min:sec" format (e.g., "5:00").
    - max_pace: Ending pace as a string in "min:sec" format (e.g., "10:00").
    - step: Step size for the ticks as a string in "min:sec" format (e.g., "0:30").
    - axis: The axis to apply the labels to ("x" or "y").
    """
    # Convert parameters to timedelta and calculate tick positions
    min_seconds = pd.to_timedelta(f"00:{min_pace}").total_seconds()
    max_seconds = pd.to_timedelta(f"00:{max_pace}").total_seconds()
    step_seconds = pd.to_timedelta(f"00:{step}").total_seconds()
    ticks = list(range(int(min_seconds), int(max_seconds) + 1, int(step_seconds)))

    # Apply to the correct axis
    if axis == "y":
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x // 60)}:{int(x % 60):02d}"))
    elif axis == "x":
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x // 60)}:{int(x % 60):02d}"))
    else:
        raise ValueError("Axis must be 'x' or 'y'.")


def plot_colorline(x, y, colors, ax=None, **kwargs):
    """
    Plot a line with colored segments.
    
    Parameters:
    x : list or array
        X coordinates of the data points.
    y : list or array
        Y coordinates of the data points.
    colors : list of tuples
        List of color tuples (e.g., (r, g, b) or (r, g, b, a)) for each segment.
    **kwargs : keyword arguments
        Additional arguments to pass to the plt.plot function (e.g., lw, ms, etc.).
    """
    if len(x) != len(y) or len(x) != len(colors):
        raise ValueError("The lengths of x, y, and colors must be the same.")
    
    if ax is None:
        ax = plt.gca()
            
    for i in range(len(x) - 1):
        ax.plot(x[i:i+2], y[i:i+2], color=colors[i], **kwargs)


def create_scalarmappable(colormap, data, vminmax=None, npoints=256):
    """
    Creates a ScalarMappable and corresponding colors for given data.

    Parameters:
    colormap (str or Colormap): Name of the colormap to use or a Colormap object.
    data (array-like): Data to be mapped to colors.
    vminmax (tuple, optional): Tuple specifying (vmin, vmax) for normalization. If None, use min and max of data.

    Returns:
    colors (array-like): Colors corresponding to the data values.
    smap (matplotlib.cm.ScalarMappable): ScalarMappable object.
    """
    if vminmax is None:
        vmin, vmax = np.min(data), np.max(data)
    else:
        vmin, vmax = vminmax

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax) # norm = plt.Normalize(vmin=vmin, vmax=vmax)
    
    if isinstance(colormap, str):
        cmap = plt.get_cmap(colormap)
    else:
        cmap = colormap
        
    colors = cmap(norm(data))
    smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # Recent change
    smap.set_array(np.linspace(vmin, vmax, npoints)) #smap.set_array([])
    
    return colors, smap


from selenium import webdriver
from selenium.webdriver.firefox.options import Options
import time


def map_convert_html2img(map_save_str, window_size):
    # Set up headless Firefox
    options = Options()
    options.add_argument("--headless")  # Run in headless mode
    driver = webdriver.Firefox(options=options)
    # Open the HTML file and take a screenshot
    
    abs_path = f"file:///{os.path.abspath(map_save_str)}" 
    driver.get(abs_path)
    

    # Apply a global scale to the map
    driver.execute_script("""
        document.body.style.transform = 'scale(2)';  /* Scale up 1.5x */
        document.body.style.transformOrigin = '0 0';   /* Adjust origin to top-left */
    """)
    #driver.set_window_size(900 * factor * 2, 900 * factor * 2)
    driver.set_window_size(*window_size)
    time.sleep(1)

    driver.save_screenshot(map_save_str.replace(".html", ".jpg"))
    driver.quit()
