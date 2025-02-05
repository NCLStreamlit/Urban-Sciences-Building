import streamlit as st
import pandas as pd
import json
import urllib.request
from urllib.parse import quote
import datetime as dt
import numpy as np
from urllib.error import URLError
import time

def search(metric, floors = None):
    """
    Fetches sensor entity data from the Urban Observatory API for a given metric and optional floor levels.

    Parameters:
        metric (str): The metric to search for (e.g., temperature, humidity).
        floors (list, optional): List of floors to filter the search by. Default is None.

    Returns:
        pd.DataFrame: A DataFrame containing the retrieved sensor data.
    """

    metric = quote(metric)
    callBase = 'https://api.usb.urbanobservatory.ac.uk/api/v2/sensors/entity'
    df_entities = pd.DataFrame()

    for floor in floors:
        floorCall = callBase + f'/?metric="{metric}&meta:buildingFloor={floor}&pageSize=100'
        for _ in range(3): 
            try:
                entities_data = json.loads(
                    urllib
                    .request
                    .urlopen(floorCall)
                    .read()
                    .decode('utf-8')
                    )
            except URLError as e: 
                print(f"Retrying due to: {e}")
                time.sleep(2) 

        for item in entities_data['items']:

            variable_name = item['feed'][0].get('metric')
            building_floor = item.get('meta').get('buildingFloor')
            room_name = item.get('meta').get('roomNumber')
            broker_name = item['feed'][0]['brokerage'][0]['broker'].get('name')
            timeseries_ID = item['feed'][0]['timeseries'][0].get('timeseriesId')
            units = item['feed'][0]['timeseries'][0]['unit'].get('name')

            df_dict = {"variable": str(variable_name),
                "floor": str(building_floor),
                "room": str(room_name),
                "broker": str(broker_name),
                "units": str(units),
                "timeseries_ID" : str(timeseries_ID)
            }

            df = pd.DataFrame(data = df_dict, index=[0])

            df_entities = pd.concat([df_entities, df], ignore_index=True)


    return df_entities


def send_request(Id, start_date, end_date, r_flag = False):
    """
    Fetches historic sensor data from the Urban Observatory API for a given sensor ID and time range.

    Parameters:
        Id (str): The sensor ID to query.
        start_date (datetime): The start date of the requested data range.
        end_date (datetime): The end date of the requested data range.
        r_flag (bool, optional): Recursion flag to prevent infinite loops. Default is False.

    Returns:
        list: A list of historical sensor data values or an empty list if the request fails.
    """

    call_base = f'https://api.usb.urbanobservatory.ac.uk/api/v2/sensors/timeseries/{Id}'
    call_var = call_base + '/historic/?startTime=' +\
    start_date.isoformat().replace('+00:00', 'Z') + '&endTime=' + end_date.isoformat().replace('+00:00', 'Z')
    print(start_date, end_date)
    for _ in range(3): 
        try:
            request = urllib.request.urlopen(call_var, timeout=5).read().decode('utf-8')
            response = json.loads(request)['historic']['values']
            
            return response

        except URLError as e: 
            print(f"Unexpected error: {e}")
            time.sleep(1.5)
        
        
            if hasattr(e, 'code'): 
                if (e.code==413) and (not r_flag):
                    print("Request too large, retrying with smaller time intervals...")

                    # Split the request into smaller time periods (e.g., 5-day batches)
                    new_start_dates, new_end_dates = get_sampling_period(start_date, end_date, batch_days=5)
                    all_responses = []

                    for new_start, new_end in zip(new_start_dates, new_end_dates):
                        all_responses.extend(send_request(Id, new_start, new_end, r_flag=True)) # Only recur once.
                    return all_responses
        


        return []


def get_sampling_period(start, end, batch_days=10):
    """
    Splits a given time range into smaller periods of specified batch size.

    Parameters:
        start (datetime): The start time of the full range.
        end (datetime): The end time of the full range.
        batch_days (int, optional): The maximum number of days to request. Default is 25.

    Returns:
        tuple: Two lists containing the start and end dates of each batch.
    """
    
    time_delta = end-start
    num_days = time_delta.days

    periods = int(np.ceil(num_days / batch_days)) + 1
    
    start = start.strftime('%Y-%m-%d %H:%M:%S')
    end = end.strftime('%Y-%m-%d %H:%M:%S')

    dates = pd.date_range(start,
                            end, 
                            periods=periods)

    starting_dates = dates[:-1]
    closing_dates = dates[1:] 

    return starting_dates, closing_dates


def format_dataframe(data):
    """
    Formats sensor data into a structured DataFrame with timestamps, resampling, and additional date columns.

    Parameters:
        data (list of dicts): List of sensor data records, each containing 'time' and 'value' fields.

    Returns:
        pd.DataFrame: A formatted DataFrame with resampled values and additional time-based columns.
    """

    if len(data) > 0:
        # Convert list of dictionaries into a Pandas DataFrame, excluding 'duration' column if present
        df = pd.DataFrame.from_records(data, exclude=['duration'])

        # Rename 'time' column to 'Timestamp' for clarity
        df.rename(columns={'time':'Timestamp'}, inplace = True)

        # Convert 'Timestamp' column to datetime format and set it as the index
        df.index = pd.to_datetime(df["Timestamp"])
        df = df.drop(columns="Timestamp")

        # Resample data to 30-minute intervals, computing the mean for each period
        df = df[["value"]].resample("30min", label = 'right').mean()

        # Add formatted timestamp columns for better readability
        df['Timestamp'] = df.index.strftime('%d/%m/%Y %H:%M')
        df['Date'] = df.index.strftime('%d/%m/%Y')
        df['Day'] = df.index.day_name()
        df['Time'] = df.index.strftime('%H:%M')

        # Reorder columns for CSV saving
        df = df[['Timestamp', 'Date', 'Day','Time', 'value']]
    else:
        print('Missing data')

    return df
  

@st.cache_resource
def timeseries_to_df(timeseries_ID, start, end):
    """
    Fetches time-series data for a given sensor ID and time range, 
    splits it into smaller periods if necessary, and formats it into a DataFrame.

    Parameters:
        timeseries_ID (str): The unique identifier of the time-series sensor.
        start (datetime): The start datetime for the data request.
        end (datetime): The end datetime for the data request.

    Returns:
        pd.DataFrame: A formatted DataFrame containing the requested time-series data.
    """

    starting_dates, closing_dates = get_sampling_period(start, end)

    with st.status("Loading data..."):
        USB_data = []

        for rng in zip(starting_dates, closing_dates):

            st.write(f"Loading {dt.datetime.strftime(rng[0], '%Y/%m/%d')} to {dt.datetime.strftime(rng[1], '%Y/%m/%d')}...")
            USB_data.extend(send_request(timeseries_ID, rng[0], rng[1]))

        df_period = format_dataframe(USB_data)
        
        return df_period


def _enforce_single_selection(edited_df):
    """
    Ensures that only one row in the DataFrame has the "Select" column set to True.
    If multiple rows are selected, only the most recently selected row is kept.

    Parameters:
        edited_df (pd.DataFrame): A DataFrame containing a "Select" column with boolean values.

    Returns:
        pd.DataFrame: The modified DataFrame with only one row selected.
    """

    # Check if more than one row is selected
    if edited_df["Select"].sum() > 1:

        # Find the last row where "Select" was True and keep it
        last_selected_idx = edited_df.index[edited_df["Select"]].tolist()[-1]

        edited_df["Select"] = False
        edited_df.at[last_selected_idx, "Select"] = True

        st.warning("Only one entity can be selected at a time.")
    return edited_df


def dataframe_with_selections(df: pd.DataFrame, init_value: bool = False) -> pd.DataFrame:
    """
    Creates a Streamlit data editor where users can select a single row at a time.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        init_value (bool, optional): Default selection state for the "Select" column. Defaults to False.
    
    Returns:
        pd.DataFrame: A filtered DataFrame containing only the selected row(s), 
                      with the "Select" column removed.
    """

     # Create a copy of the DataFrame and insert a "Select" checkbox column at the start
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", init_value)

    # Get dataframe row-selections from user with st.data_editor
    edited_df = st.data_editor(
        df_with_selections,
        hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
        disabled=df.columns,
    )

    # Ensure only one row is selected at a time
    edited_df = _enforce_single_selection(edited_df)

    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Select]

    return selected_rows.drop('Select', axis=1)


# --- Streamlit App ---
st.title('Urban Science Building Explorer')

# Documentation Links
doc_url = "https://api.usb.urbanobservatory.ac.uk/"
ubs_3d_url = "https://3d.usb.urbanobservatory.ac.uk/"

st.components.v1.iframe(ubs_3d_url, height=400, scrolling=True)
st.write("Click [here](%s) for a fullscreen view." % ubs_3d_url)
st.write("Find the full documentation [here](%s)." % doc_url)


# Step 1: Search Database
st.divider()
with st.container(border=False):
    st.header('Step 1: Search Database') 

    search_term = st.selectbox("Select building floor", options=("CO 2", 
                                                          "Room Temperature", 
                                                          "Room Brightness", 
                                                          "Room Occupied", 
                                                          "Power Level", 
                                                          "Total Energy", 
                                                          "Window Position")
    )

    floor_select = st.selectbox("Select building floor", 
                                options=("G","1","2","3","4","5","6"))
    
    df_entities = search(search_term, floors=floor_select)
    df_selection = dataframe_with_selections(df_entities)
    st.write("Your selection:")
    st.write(df_selection)


if not df_selection.empty: 
    # Step 2: Load Data
    st.divider()
    with st.container(border=False): 
        st.header('Step 2: Load Data')  
        st.write('This may take a while depending on the size of date range.')
        
        timeseries_ID = df_selection['timeseries_ID'].values[0]
        entity_units = df_selection['units'].values[0]
        entity_name = df_selection['variable'].values[0]
        entity_location = f"Floor: {df_selection['floor'].values[0]}, Room: {df_selection['room'].values[0]}"

        start_date = st.date_input("Start Date", 
                                value=dt.datetime.strptime('2025/01/01', '%Y/%m/%d'),
                                min_value=None,
                                max_value=None)
        
        end_date = st.date_input("End Date", value='today',
                                min_value=start_date, max_value=None)
        
        data = timeseries_to_df(timeseries_ID, start_date, end_date)
        st.write(data)


    # Step 3: Plot Data
    st.divider()
    with st.container(border=False): 
        st.header('Step 3: Plotting')

        chart_types = {
        "Scatter": st.scatter_chart,
        "Bar": st.bar_chart,
        "Line": st.line_chart,
        "Area": st.area_chart,
        }

        chart_select = st.selectbox("Type", options=chart_types.keys())
        y_label = f"{entity_name} [{entity_units}]"
        st.subheader(entity_location)
        #Presents x-ticks better
        data = data.drop(columns=['Timestamp', 'Date', 'Day', 'Time'])
        chart_types[chart_select](data=data, y_label=y_label)


