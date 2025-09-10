import pandas as pd
import numpy as np
import plotly.express as px
import math
from scipy.stats import pearsonr
from datetime import datetime

# Define lists of airfoil codes
all_airfoils = ['0012', '0015', '0018',
                '4212', '4215', '4218',
                '4512', '4812']

symmetric_airfoils = ['0012', '0015', '0018']
non_symmetric_airfoils = ['4212', '4215', '4218']
cambered_airfoils = ['4212', '4512', '4812']

# Combine symmetric and non-symmetric airfoils for thickness analysis
thickness_airfoils = symmetric_airfoils + non_symmetric_airfoils

# Define color codes for each group
symmetric_colors = ['#00007f', '#0000ff', '#00aaff']
non_symmetric_colors = ['#005500', '#55aa7f', '#008900']
cambered_colors = ['#ffaa00', '#aa0000', '#ff0000']

# Create dictionaries mapping airfoils to colors
symmetric_color_dict = dict(zip(symmetric_airfoils, symmetric_colors))
non_symmetric_color_dict = dict(zip(non_symmetric_airfoils, non_symmetric_colors))

cambered_color_dict = dict(zip(cambered_airfoils, cambered_colors))

# Merge dictionaries for full thickness-based color mapping
thickness_color_dict = {**symmetric_color_dict, **non_symmetric_color_dict}


#For numerical results
def read_xflr5(filename):
    """
    Reads XFLR5 aerodynamic data and processes key coefficients.

    Parameters:
        filename (str): Path to the .csv file exported from XFLR5.

    Returns:
        pandas.DataFrame: Filtered aerodynamic data with additional Cl/Cd ratio.
    """
    df = pd.read_csv(
        filename,
        skiprows=10,
        names=["alpha", "CL", "CD", "CDp", "Cm", "Top Xtr", "Bot Xtr", "Cpmin", "Chinge", "XCp"]
    )

    # Filter angles between 0 and 12 degrees, even numbers only
    df_filtered = df[
        (df['alpha'].between(0, 12)) &
        (df['alpha'] % 2 == 0)
        ].copy()

    # Calculate Lift-to-Drag Ratio
    df_filtered["CL/CD"] = df_filtered["CL"] / df_filtered["CD"]

    return df_filtered

def read_xflr5_full(filename):
    """
    Reads full-range XFLR5 aerodynamic data with extended angle coverage.

    Parameters:
        filename (str): Path to the .csv file exported from XFLR5.

    Returns:
        pandas.DataFrame: Aerodynamic data filtered between -5° and 15° with Cl/Cd ratio calculated.
    """
    df = pd.read_csv(
        filename,
        skiprows=10,
        names=["alpha", "CL", "CD", "CDp", "Cm", "Top Xtr", "Bot Xtr", "Cpmin", "Chinge", "XCp"]
    )

    # Keep angles between -5° and 15°
    df_filtered = df[df['alpha'].between(-5, 15)].copy()

    # Calculate Lift-to-Drag Ratio
    df_filtered["CL/CD"] = df_filtered["CL"] / df_filtered["CD"]

    # Drop unnecessary columns for focused analysis
    df_filtered = df_filtered.drop(
        columns=['CDp', 'Cm', 'Top Xtr', 'Bot Xtr', 'Cpmin', 'Chinge', 'XCp']
    )

    return df_filtered

def get_alpha_at_max_cl_xflr5(airfoil_code, reynolds_number):
    filename = f'NACA {airfoil_code}_T1_Re0.{str(reynolds_number)[:3]}_M0.03_N9.0.csv'
    data = read_xflr5(filename)
    return data.loc[data['CL'].idxmax(), 'alpha']

def get_lift_curve_slope_xflr5(airfoil_code, reynolds_number):
    filename = f'NACA {airfoil_code}_T1_Re0.{str(reynolds_number)[:3]}_M0.03_N9.0.csv'
    data = read_xflr5(filename)

    alpha_max_cl = get_alpha_at_max_cl_xflr5(airfoil_code, reynolds_number)

    cl_before = data.loc[data['alpha'] == alpha_max_cl - 2.0, 'CL'].values[0]
    cl_at_2deg = data.loc[data['alpha'] == 2.0, 'CL'].values[0]

    slope = (cl_before - cl_at_2deg) / ((alpha_max_cl - 4.0) * math.pi / 180)
    return slope

def get_maximum_cl_xflr5(airfoil_code, reynolds_number):
    filename = f'NACA {airfoil_code}_T1_Re0.{str(reynolds_number)[:3]}_M0.03_N9.0.csv'
    data = read_xflr5(filename)
    return max(data['CL'])

def get_maximum_cl_cd_xflr5(airfoil_code, reynolds_number):
    filename = f'NACA {airfoil_code}_T1_Re0.{str(reynolds_number)[:3]}_M0.03_N9.0.csv'
    data = read_xflr5(filename)
    return max(data['CL/CD'])

def get_minimum_cl_xflr5(airfoil_code, reynolds_number):
    filename = f'NACA {airfoil_code}_T1_Re0.{str(reynolds_number)[:3]}_M0.03_N9.0.csv'
    data = read_xflr5(filename)
    return min(data['CL'])

def get_minimum_cl_cd_xflr5(airfoil_code, reynolds_number):
    filename = f'NACA {airfoil_code}_T1_Re0.{str(reynolds_number)[:3]}_M0.03_N9.0.csv'
    data = read_xflr5(filename)
    return min(data['CL/CD'])

# For experimental results
def drag_coefficient(drag_force, velocity):
    return 2 * drag_force / (1.2 * 0.3 * 0.15 * velocity ** 2)

def lift_coefficient(lift_force, velocity):
    return 2 * lift_force / (1.2 * 0.3 * 0.15 * velocity ** 2)

def clean_experiment_data(airfoil_code, test_velocity):
    airfoil_code = str(airfoil_code)
    file_name = f'{airfoil_code} v={test_velocity}.csv'

    data = pd.read_csv(file_name)
    data.columns = data.iloc[0]
    data = data[1:].reset_index(drop=True)

    data = data.iloc[:, :7]
    data = data.drop(columns=['Time\n(h:min:sec)'])

    data['Act.AirSpeed\n(m/s)'] = data['Act.AirSpeed\n(m/s)'].astype(float)
    mean_air_speed = data['Act.AirSpeed\n(m/s)'].mode()[0]

    data = data[(data['Act.AirSpeed\n(m/s)'] < mean_air_speed + 0.5) &
                (data['Act.AirSpeed\n(m/s)'] > mean_air_speed - 0.5)]

    data['DragForce\n(N)'] = data['DragForce\n(N)'].astype(float) * (-1)
    data['LiftForce\n(N)'] = data['LiftForce\n(N)'].astype(float)

    data['LD'] = data['LiftForce\n(N)'] / data['DragForce\n(N)']
    data['Act.Angle\n(deg)'] = data['Act.Angle\n(deg)'].astype(float)

    # Filter for even angles of attack below 14 degrees
    data = data[data['Act.Angle\n(deg)'] % 2 == 0]
    data = data[data['Act.Angle\n(deg)'] < 14]

    data['aoa'] = data['Act.Angle\n(deg)']  # Angle of attack

    data['CD'] = round(drag_coefficient(data['DragForce\n(N)'], test_velocity), 6)
    data['CL'] = round(lift_coefficient(data['LiftForce\n(N)'], test_velocity), 6)

    data['lift_force'] = data['LiftForce\n(N)']
    data['lift_to_drag_ratio'] = data['LD']

    return data

def group_coefficient(coef_type, airfoil_code, test_velocity):
    data = clean_experiment_data(airfoil_code, test_velocity)
    return data.groupby('aoa')[coef_type].apply(lambda x: x.mode()[0]).reset_index()

def get_alpha_at_max_cl_exp(model, v):
    return group_coefficient('CL', model, v).loc[group_coefficient('CL', model, v)['CL'].idxmax(), 'aoa']

def get_lift_curve_slope_exp(model, v):
    ex = clean_experiment_data(model, v)
    return (ex.loc[ex['aoa'] == get_alpha_at_max_cl_exp(model, v) - 2.0, 'CL'].values[0] -
            ex.loc[ex['aoa'] == 2.0, 'CL'].values[0]) / ((get_alpha_at_max_cl_exp(model, v) - 4) * math.pi / 180)

def get_maximum_cl_exp(model, v):
    return max(group_coefficient('CL', model, 12)['CL'])

def get_maximum_cl_cd_exp(model, v):
    return max(group_coefficient('LD', model, 12)['LD'])

def get_minimum_cl_exp(model, v):
    return min(group_coefficient('CL', model, 12)['CL'])

def get_minimum_cl_cd_exp(model, v):
    return min(group_coefficient('LD', model, 12)['LD'])

# Defining the graph scale for characteristic analysis for both numerical and experimental results
maxldthickness = [get_maximum_cl_cd_xflr5(i, 119000) for i in thickness_airfoils] + [get_maximum_cl_cd_exp(i, 12) for i in thickness_airfoils]
mmaxldthickness = max(maxldthickness)
mminldthickness = min(maxldthickness)

maxlthickness = [get_maximum_cl_xflr5(i, 119000) for i in thickness_airfoils] + [get_maximum_cl_exp(i, 12) for i in thickness_airfoils]
mmaxlthickness = max(maxlthickness)
mminlthickness = min(maxlthickness)

maxslopelthickness = [get_lift_curve_slope_xflr5(i, 119000) for i in thickness_airfoils] + [get_lift_curve_slope_exp(i, 12) for i in thickness_airfoils]
mmaxslopelthickness = max(maxslopelthickness)
mminslopelthickness = min(maxslopelthickness)

maxldcambered = [get_maximum_cl_cd_xflr5(i, 119000) for i in cambered_airfoils] + [get_maximum_cl_cd_exp(i, 12) for i in cambered_airfoils]
mmaxldcambered = max(maxldcambered)
mminldcambered = min(maxldcambered)

maxlcambered = [get_maximum_cl_xflr5(i, 119000) for i in cambered_airfoils] + [get_maximum_cl_exp(i, 12) for i in cambered_airfoils]
mmaxlcambered = max(maxlcambered)
mminlcambered = min(maxlcambered)

maxslopelcambered = [get_lift_curve_slope_xflr5(i, 119000) for i in cambered_airfoils] + [get_lift_curve_slope_exp(i, 12) for i in cambered_airfoils]
mmaxslopelcambered = max(maxslopelcambered)
mminslopelcambered = min(maxslopelcambered)

# Defining graph scale for alpha analysis for both numerical and experimental results
maxld = max([get_maximum_cl_cd_xflr5(i, 119000) for i in all_airfoils] + [
        get_maximum_cl_cd_exp(i, 12) for i in all_airfoils])

maxl = max([get_maximum_cl_xflr5(i, 119000) for i in all_airfoils] + [
        get_maximum_cl_exp(i, 12) for i in all_airfoils])

minld = min([get_minimum_cl_cd_xflr5(i, 119000) for i in all_airfoils] + [
        get_minimum_cl_cd_exp(i, 12) for i in all_airfoils])

minl = min([get_minimum_cl_xflr5(i, 119000) for i in all_airfoils] + [
        get_minimum_cl_exp(i, 12) for i in all_airfoils])

#Plot the alpha analysis
def plot_coefficient_graph(test_velocity, color_map, airfoil_list, output_filename, bys):
    graph_title = ""
    all_data_frames = []

    coefficient_types = [bys]

    for coef_type in coefficient_types:
        if coef_type == 'CD':
            graph_title = 'Drag Coefficient (Cd)'
        elif coef_type == 'CL':
            graph_title = 'Lift Coefficient (Cl)'
        elif coef_type == 'LD':
            graph_title = 'Lift-to-Drag Ratio (Cl/Cd)'

        for airfoil in airfoil_list:
            grouped_data = group_coefficient(coef_type, airfoil, test_velocity)
            grouped_data['Airfoil'] = airfoil
            all_data_frames.append(grouped_data)

        all_data = pd.concat(all_data_frames, ignore_index=True)

        fig = px.scatter(
            all_data,
            x='aoa',
            y=coef_type,
            color='Airfoil',
            color_discrete_map=color_map
        )

        fig.update_traces(mode='markers+lines', line_shape='linear')

        # Update layout with thicker, larger, bold title
        fig.update_layout(
            title={
                'text': f'<b>{graph_title}</b>',  # Bold title
                'x': 0.5,  # Center alignment
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24}  # Large font size for visibility
            },
            xaxis_title={
                'text': 'Angle of Attack (°)',
                'font': {'size': 18, 'family': 'Arial'}
            },
            yaxis_title={
                'text': '',
                'font': {'size': 18, 'family': 'Arial'}
            },
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Arial', size=14),
            xaxis=dict(
                showgrid=True,
                gridcolor='lightgrey',
                zerolinecolor='grey',
                linecolor='grey',
                minor=dict(showgrid=True),
                tickfont=dict(family='Arial', size=16)  # Bigger tick labels
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='lightgrey',
                zerolinecolor='grey',
                linecolor='grey',
                minor=dict(showgrid=True),
                tickfont=dict(family='Arial', size=16)  # Bigger tick labels
            ),
            legend_title_font=dict(size=16),
            legend_font=dict(size=14)
        )

        fig.write_image(f'{output_filename}.pdf', scale=2, width=1000, height=500)


# all in one function for … vs. Alpha
def make_xfalpha(result, variables, by):
    global df, xvar, y_column, color_map, show, t, range_y, y_title, titlet
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f'{result}{by}{now}.pdf'
    airfoil_list = ""


    all_data_frames = []
    bys = [by]

    if variables == 'thickness':
        color_map = thickness_color_dict
        airfoil_list = thickness_airfoils

    elif variables == 'cambered':
        color_map = cambered_color_dict
        airfoil_list = cambered_airfoils

    for coef_type in bys:
        if coef_type == 'CD':
            y_title = 'Drag Coefficient (CD)'
            y_column = 'CD'
            t = "(g)"

        elif coef_type == 'CL':
            y_title = 'Lift Coefficient (CL)'
            y_column = 'CL'
            t = "(a)"
            show = False
            if variables == 'thickness':
                range_y = dict(range=[-1, 2])
            elif variables == 'cambered':
                range_y = dict(range=[-0.5, 2])

        elif coef_type == 'LD':
            y_title = 'Lift-to-Drag Ratio (L/D)'
            t = "(b)"
            show = True
            if variables == 'thickness':
                range_y = dict(range=[-45, 65])
            elif variables == 'cambered':
                range_y = dict(range=[-20, 70])

        for airfoil in airfoil_list:
            if result == 'XFLR5':
                filename = f'NACA {airfoil}_T1_Re0.119_M0.03_N9.0.csv'
                df = read_xflr5_full(filename)
                xvar = 'alpha'
                if coef_type == 'LD':
                    y_column = 'CL/CD'

            elif result == 'Experiment':
                df = group_coefficient(coef_type, airfoil, test_velocity=12)
                xvar = 'aoa'
                if coef_type == 'LD':
                    y_column = 'LD'

            df['Airfoil'] = airfoil
            all_data_frames.append(df)

        if result == 'Experiment':
            titlet = 'Experimental result'
        elif result == 'XFLR5':
            titlet = 'Numerical simulation'

        all_data = pd.concat(all_data_frames, ignore_index=True)

        fig = px.scatter(
            all_data,
            x=xvar,
            y=y_column,
            color='Airfoil',
            color_discrete_map=color_map
        )

        fig.update_traces(mode='markers+lines', line_shape='spline')

        fig.update_layout(
            title={
                'text': titlet,
                'font': {'size': 22, 'family': 'Arial', 'color': 'black'},
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title={
                'text': 'Angle of Attack (°)',
                'font': {'size': 18, 'family': 'Arial', 'color': 'black'}
            },
            yaxis_title={
                'text': y_title,
                'font': {'size': 18, 'family': 'Arial', 'color': 'black'}
            },
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Arial', size=14),
            xaxis=dict(
                showgrid=True,
                gridcolor='lightgrey',
                zerolinecolor='grey',
                linecolor='grey',
                minor=dict(showgrid=True),
                tickfont=dict(family='Arial', size=16, color='black'),
                mirror=True,
                linewidth=2
            ),

            yaxis=dict(
                showgrid=True,
                gridcolor='lightgrey',
                zerolinecolor='grey',
                linecolor='grey',
                minor=dict(showgrid=True),
                tickfont=dict(family='Arial', size=16, color='black'),
                mirror=True,
                linewidth=2,
                **range_y
            ),

            width=1000,
            height=500,

            legend_title_font=dict(size=16),
            annotations=[
                dict(
                    text=t,
                    xref='paper', yref='paper',
                    x=0.95, y=0.1,
                    xanchor='right', yanchor='bottom',
                    font=dict(size=26, family='Arial', color='black'),
                    showarrow=False
                )
            ],
            legend=dict(
                orientation='v',
                itemwidth=50,
                traceorder='normal',
                bgcolor='rgba(0,0,0,0)',
                borderwidth=1
            ),
            showlegend=show,
        )

        fig.update_xaxes(
            range=[-6, 16]
        )

        fig.show()
        fig.write_image(f'{output_filename}.pdf', scale=2, width=1000, height=500)

#Plot the characteristic graph
def make_character_graph(results, variables, y_data):
    global yaxis_range_l, yaxis_range_ld, yaxis_range_slopel, fig, cambered_data, thickness_data
    reynolds_number = 119000
    tested_velocity = 12
    save = 1
    graph_name_y_data = {'LD': '<br>Maximum lift-to-drag ratio',
                         'Lift_Slope': 'Lift Slope',
                         'Max_CL': '<br>Maximum Lift Coefficient',
                         'Stall_Alpha': 'Stall Angle'}
    y_axis_title_y_data = {'LD': 'Maximum lift-to-drag ratio',
                           'Lift_Slope': 'Lift Slope',
                           'Max_CL': 'Max CL',
                           'Stall_Alpha': 'Stall Angle'}
    if results == 'XFLR5':
        # Symmetrical DataFrame
        symmetric_data = pd.DataFrame({
            't/C': [int(n[2:]) for n in symmetric_airfoils],
            'LD': [get_maximum_cl_cd_xflr5(n, reynolds_number) for n in symmetric_airfoils],
            'Lift_Slope': [get_lift_curve_slope_xflr5(n, reynolds_number) for n in symmetric_airfoils],
            'Max_CL': [get_maximum_cl_xflr5(n, reynolds_number) for n in symmetric_airfoils],
            'Stall_Alpha': [get_alpha_at_max_cl_xflr5(n, reynolds_number) for n in symmetric_airfoils]
        })
        # Non-Symmetrical DataFrame
        non_symmetric_data = pd.DataFrame({
            't/C': [int(n[2:]) for n in non_symmetric_airfoils],
            'LD': [get_maximum_cl_cd_xflr5(n, reynolds_number) for n in non_symmetric_airfoils],
            'Lift_Slope': [get_lift_curve_slope_xflr5(n, reynolds_number) for n in non_symmetric_airfoils],
            'Max_CL': [get_maximum_cl_xflr5(n, reynolds_number) for n in non_symmetric_airfoils],
            'Stall_Alpha': [get_alpha_at_max_cl_xflr5(n, reynolds_number) for n in non_symmetric_airfoils]
        })
        # Add Airfoil Type
        symmetric_data['Airfoil type'] = 'Symmetrical Airfoils'
        non_symmetric_data['Airfoil type'] = 'Non-Symmetrical Airfoils'

        # Combine
        thickness_data = pd.concat([non_symmetric_data, symmetric_data])

        # Cambered DataFrame
        cambered_data = pd.DataFrame({
            'Xz/C': [float(f'0.{n[1]}') for n in cambered_airfoils],
            'LD': [get_maximum_cl_cd_xflr5(n, reynolds_number) for n in cambered_airfoils],
            'Lift_Slope': [get_lift_curve_slope_xflr5(n, reynolds_number) for n in cambered_airfoils],
            'Max_CL': [get_maximum_cl_xflr5(n, reynolds_number) for n in cambered_airfoils],
            'Stall_Alpha': [get_alpha_at_max_cl_xflr5(n, reynolds_number) for n in cambered_airfoils]
        })

    elif results == 'Experiment':
        # Symmetrical DataFrame
        symmetric_data = pd.DataFrame({
            't/C': [int(n[2:]) for n in symmetric_airfoils],
            'LD': [get_maximum_cl_cd_exp(n, tested_velocity) for n in symmetric_airfoils],
            'Lift_Slope': [get_lift_curve_slope_exp(n, tested_velocity) for n in symmetric_airfoils],
            'Max_CL': [get_maximum_cl_exp(n, tested_velocity) for n in symmetric_airfoils],
            'Stall_Alpha': [get_alpha_at_max_cl_exp(n, tested_velocity) for n in symmetric_airfoils]
        })
        # Non-Symmetrical DataFrame
        non_symmetric_data = pd.DataFrame({
            't/C': [int(n[2:]) for n in non_symmetric_airfoils],
            'LD': [get_maximum_cl_cd_exp(n, tested_velocity) for n in non_symmetric_airfoils],
            'Lift_Slope': [get_lift_curve_slope_exp(n, tested_velocity) for n in non_symmetric_airfoils],
            'Max_CL': [get_maximum_cl_exp(n, tested_velocity) for n in non_symmetric_airfoils],
            'Stall_Alpha': [get_alpha_at_max_cl_exp(n, tested_velocity) for n in non_symmetric_airfoils]
        })
        # Add Airfoil Type
        symmetric_data['Airfoil type'] = 'Symmetrical Airfoils'
        non_symmetric_data['Airfoil type'] = 'Non-Symmetrical Airfoils'

        # Combine
        thickness_data = pd.concat([non_symmetric_data, symmetric_data])

        # Cambered DataFrame
        cambered_data = pd.DataFrame({
            'Xz/C': [float(f'0.{n[1]}') for n in cambered_airfoils],
            'LD': [get_maximum_cl_cd_exp(n, tested_velocity) for n in cambered_airfoils],
            'Lift_Slope': [get_lift_curve_slope_exp(n, tested_velocity) for n in cambered_airfoils],
            'Max_CL': [get_maximum_cl_exp(n, tested_velocity) for n in cambered_airfoils],
            'Stall_Alpha': [get_alpha_at_max_cl_exp(n, tested_velocity) for n in cambered_airfoils]
        })
    # elif results == 'both':

    if variables == 'thickness':
        fig = px.scatter(
            thickness_data,
            width=600,
            height=400,
            x='t/C',
            y=y_data,
            color='Airfoil type',
            symbol='Airfoil type',
            symbol_sequence=['square', 'diamond'],
        )
        fig.update_traces(opacity=0.8)

        fig.update_layout(
            xaxis_title='t/C',
            xaxis=dict(
                tickmode='array',
                tickvals=[12, 15, 18],
                ticktext=['12', '15', '18']
            ),
            xaxis_title_font=dict(
                family='Arial',
                size=26,
                color='black'
            ),
            title_font=dict(
                family="Arial",
                size=26,
                color='black'
            ),

        )

        yaxis_range_l = dict(range=[mminlthickness - 0.1, mmaxlthickness + 0.1])
        yaxis_range_ld = dict(range=[mminldthickness - 3, mmaxldthickness + 6])
        yaxis_range_slopel = dict(range=[mminslopelthickness - 1.5, mmaxslopelthickness + 0.5])

    elif variables == 'cambered':
        fig = px.scatter(
            cambered_data,
            width=500,
            height=400,
            x='Xz/C',
            y=y_data,
            color_discrete_sequence=["green"],
            symbol_sequence=['square'],
        )
        fig.update_traces(opacity=1)
        fig.update_layout(
            xaxis_title='Xzmax/C',
            xaxis=dict(
                tickmode='array',
                tickvals=[0.2, 0.5, 0.8],
                ticktext=['0.2', '0.5', '0.8'],
            ),
            title_font=dict(
                family="Arial",
                size=26,
                color='black'
            ),
            xaxis_title_font=dict(
                family='Arial',
                size=26,
                color='black'
            ),
        )
        yaxis_range_l = dict(range=[mminlcambered, mmaxlcambered])
        yaxis_range_ld = dict(range=[mminldcambered - 15, 100])
        yaxis_range_slopel = dict(range=[mminslopelcambered - 1, mmaxslopelcambered + 0.5])

    # Layout Updates
    fig.update_layout(
        yaxis_title=y_axis_title_y_data[y_data],
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            tickfont=dict(size=26,
                          family='Arial',
                          color='black'),
            linecolor='grey',
            zerolinecolor='grey'
        ),
        yaxis=dict(
            tickfont=dict(size=26,
                          family='Arial',
                          color='black'),
            zerolinecolor='grey',
            linecolor='grey'
        ),
        yaxis_title_font=dict(
            family='Arial',
            size=26,
            color='black'
        ),
        legend_font=dict(
            size=18,
            family='Arial',
            color='black'
        ),
    )

    fig.update_traces(marker=dict(size=15))

    if y_data == 'Lift_Slope':
        fig.update_layout(
            annotations=[
                dict(
                    text="(d)",
                    xref='paper', yref='paper',
                    x=1.2, y=0,
                    xanchor='right', yanchor='bottom',
                    font=dict(size=26, family='Arial', color='black'),
                    showarrow=False
                )
            ],
            showlegend=False,
            yaxis=yaxis_range_slopel,
            width=500,
            height=450,
            margin=dict(l=100, r=100, t=100, b=100)

        )

        fig.add_hline(
            y=2 * math.pi,
            line_dash='dash',
            annotation=dict(
                text='y = 2π',
                font_size=22,
                align='right',
                font=dict(color='black'),
                yshift=-35,

            )
        )

    elif y_data == 'LD':
        fig.update_layout(
            annotations=[
                dict(
                    text="(c)",
                    xref='paper', yref='paper',
                    x=1.2, y=0,
                    xanchor='right', yanchor='bottom',
                    font=dict(size=26, family='Arial', color='black'),
                    showarrow=False
                )
            ],
            showlegend=False,
            yaxis=yaxis_range_ld,
            width=500,
            height=450,
            margin=dict(l=100, r=100, t=100, b=100)
        )

    elif y_data == 'Max_CL':
        fig.update_layout(
            annotations=[
                dict(
                    text="(e)",
                    xref='paper', yref='paper',
                    x=1.2, y=0,
                    xanchor='right', yanchor='bottom',
                    font=dict(size=26, family='Arial', color='black'),
                    showarrow=False
                )
            ],
            width=500,

        )
        if variables == 'thickness':
            fig.update_layout(
                legend=dict(
                    orientation='h',
                    x=0.5,
                    y=-0.75,
                    xanchor='center',
                    yanchor='bottom',
                    xref='paper', yref='paper',

                ),
                yaxis=yaxis_range_l,
                width=500,
                height=535,
                margin=dict(l=100, r=100, t=100, b=150)
            )
        else:
            fig.update_layout(
                margin=dict(l=100, r=100, t=100, b=100),
                height=450,
            )

    elif y_data == 'Stall_Alpha':
        fig.update_layout(
            annotations=[
                dict(
                    text="(f)",
                    xref='paper', yref='paper',
                    x=1.2, y=0,
                    xanchor='right', yanchor='bottom',
                    font=dict(size=26, family='Arial', color='black'),
                    showarrow=False
                )
            ],
            showlegend=False,
            width=500,
            height=450,
            margin=dict(l=100, r=100, t=100, b=100)
        )

        fig.update_yaxes(
            tickmode='linear',
            tick0=0,
            dtick=2,
            range=[0, 13]

        )
    fig.update_layout(
        xaxis=dict(mirror=True,
                   linewidth=2
                   ),
        yaxis=dict(mirror=True,
                   linewidth=2
                   ),
        legend=dict(
            orientation='h',
            itemwidth=50,
            traceorder='normal',
            bgcolor='rgba(0,0,0,0)',
            borderwidth=1
        ),

    )
    fig.show()

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{results}{variables}_{now}.pdf"

    if (save == 1) and (y_data == 'Max_CL') and (variables == 'thickness'):
        fig.write_image(filename, scale=2, width=500, height=535)

    else:
        fig.write_image(filename, scale=2, width=500, height=450)

#Make an all-purpose graph for both types of analysis
def all_purpose_graph(results, graph_type, y_data, variables):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if graph_type == 'alpha':
        if results == 'XFLR5':
            if variables == 'thickness':
                return make_xfalpha('119000', thickness_color_dict, thickness_airfoils,
                                    f'{results}{variables}{now}', y_data)
            elif variables == 'cambered':
                return make_xfalpha('119000', cambered_color_dict, cambered_airfoils,
                                    f'{results}{variables}{now}', y_data)

        elif results == 'Experiment':
            if variables == 'thickness':
                return plot_coefficient_graph(12, thickness_color_dict, thickness_airfoils,
                                              f'{results}{variables}{now}', y_data)
            elif variables == 'cambered':
                return plot_coefficient_graph(12, cambered_color_dict, cambered_airfoils,
                                              f'{results}{variables}{now}', y_data)

    elif graph_type == 'characteristics':
        return make_character_graph(results, variables, y_data)

#Correlations
def pearson(foil, what):
    global m1
    m2 = read_xflr5(f'NACA {foil}_T1_Re0.119_M0.03_N9.0.csv')
    if what == 'LD':
        m1 = group_coefficient('LD', foil, test_velocity = 12)
        merged = pd.merge(m1, m2, left_on='aoa', right_on = 'alpha', how='inner')
        m1 = merged['LD']
        m2 = merged['CL/CD']

    elif what == 'L':
        m1 = group_coefficient('CL', foil, test_velocity = 12)
        m1 = m1.rename(columns={'CL': 'LL'})
        merged = pd.merge(m1, m2, left_on='aoa', right_on = 'alpha', how='inner')
        m1 = merged['LL']
        m2 = merged['CL']

    elif what == 'D':
        m1 = group_coefficient('CD', foil, test_velocity = 12)
        m1 = m1.rename(columns={'CD': 'DD'})
        merged = pd.merge(m1, m2, left_on='aoa', right_on = 'alpha', how='inner')
        m1 = merged['DD']
        m2 = merged['CD']

    method1 = np.array(m1)
    method2 = np.array(m2)
    r, p = pearsonr(method1, method2)
    return r