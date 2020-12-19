import datetime
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import random
import datetime
from os.path import dirname, join
from os import listdir


from .data_access import GetData
from .utils import read_yaml


def data_source(time_indicator, amount_indicator):
    try:
        source = GetData(data_query_path="sample_data.csv",
                         data_source="csv",
                         time_indicator=time_indicator,
                         feature=amount_indicator, test=1000)
        source.query_data_source()
        source.convert_feature()
        data = source.data
    except Exception as e:
        data = pd.DataFrame()
        print("no data is available")
    return data


def get_filters(results, customer_indicator, amount_indicator):
    data_pv = results.groupby(customer_indicator).agg({"prediction_values": "sum"}).reset_index()
    top_100_customers = list(data_pv.sort_values("prediction_values",ascending=False)[customer_indicator])[0:100]
    worst_100_customers = list(data_pv.sort_values("prediction_values", ascending=True)[customer_indicator])[0:100]
    calculation_selection = ['Sum Of ' + amount_indicator, 'Average Of ' + amount_indicator]
    filter_datas = [top_100_customers, worst_100_customers, calculation_selection]
    filter_ids = ['top_100_customers', 'worst_100_customers', 'calculation_type']
    filter_sizes = [100, 100, 2]
    multiple_selection = [False] * 3
    values = [top_100_customers[0], worst_100_customers[0], 'Sum Of ' + amount_indicator]
    filters = list(zip(filter_ids, filter_datas, filter_sizes, multiple_selection, values))
    return 3, filters, filter_ids, top_100_customers, worst_100_customers


def decide_time_period(date, time_period):
    if time_period == 'hour':
        return datetime.datetime.strptime(str(date)[0:16], "%Y-%m-%d %H")
    if time_period == 'day':
        return datetime.datetime.strptime(str(date)[0:10], "%Y-%m-%d")
    if time_period == 'month':
        return datetime.datetime.strptime(str(date)[0:7], "%Y-%m")
    if time_period == 'year':
        return datetime.datetime.strptime(str(date)[0:4], "%Y")
    if time_period == 'week':  # only shows mondays for weekly comparison
        week_day = datetime.datetime.strptime(str(date)[0:10], "%Y-%m-%d").isoweekday()
        return datetime.datetime.strptime(str(date)[0:10], "Y-%m-%d") - datetime.timedelta(days=week_day-1)
    if time_period == 'week':  # only shows mondays for weekly comparison
        date = datetime.datetime.strptime(str(date)[0:10], "Y-%m-%d")
        week_day = date.isoweekday()
        week = date.isocalendar()[1]
        mondays_cal_day_count = week_day-1 if week % 2 == 0 else week_day - 1 + 7
        return datetime.datetime.strptime(str(date)[0:10], "Y-%m-%d") - datetime.timedelta(days=mondays_cal_day_count)


def get_data_time_period_column(data, results, time_indicator, time_period):
    data = pd.concat([data, results])
    data[time_indicator + '_per_' + time_period] = data[time_indicator].apply(
        lambda x: decide_time_period(x, time_period))
    return data


def pivoting_data_per_time_period(data, results, time_indicator, time_period, amount_indicator, customer_indicator):
    data = get_data_time_period_column(data, results, time_indicator, time_period)
    data[amount_indicator + "_sum"], data[amount_indicator + "_mean"] = data[amount_indicator], data[amount_indicator]
    data_pv = data.groupby([customer_indicator, time_indicator + '_per_' + time_period]).agg(
        {amount_indicator + "_sum": "sum", amount_indicator + "_mean": "mean"}).reset_index()
    return data_pv


def adding_filter_to_pane(added_filters, f_style):
    return html.Div(added_filters, style=f_style)


def adding_plots_to_pane(plot_id, hover_data, size):
    return html.Div([
        dcc.Graph(
            id=plot_id,
            hoverData={'points': [hover_data]}
        )
    ], style={'width': str(size) + '%', 'display': 'inline-block', 'padding': '0 90'})


def adding_filter(filter_id, labels, size, is_multi_select, value):
    return html.Div([
        html.Div(filter_id, style={'width': '40%', 'float': 'left', 'display': 'inline-block'}),
        dcc.Dropdown(
            id=filter_id,
            options=[{'label': i, 'value': i} for i in labels],
            multi=True if is_multi_select else False,
            value=value
        )
    ],
        style={'width': str(size) + '%', 'display': 'inline-block'})


"""
Top 100 customer of Time Period Change
Worst 100 Customer Of Time Period Value Change
Time Line Chart with filter selected average/Total value per time period with predicted life time value
One Single Value Of Total Life Time Value Filter selected.
Comparison Of User With Selected Time Period And their Predicted Values

"""


def get_hover_data(data, time_indicator, time_period, number_of_graph):
    return [{'customdata': sorted(list(data[time_indicator + '_per_' + time_period]))[-3]}] * number_of_graph


def create_dashboard(server, customer_indicator, amount_indicator, directory, time_indicator, time_period):
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(__name__, server=server, external_stylesheets=external_stylesheets, routes_pathname_prefix='/dash/')
    try:
        data = data_source()
        results = get_results(directory)
        num_f_p, filters, filter_ids, top_100_customers, worst_100_customers = get_filters(results,
                                                                                                        customer_indicator,
                                                                                                        amount_indicator)
        app.layout = html.Div()
        if len(data) == 0:
            return app
    except Exception as e:
        print(e)
        app.layout = html.Div()
        return app
    data = pivoting_data_per_time_period(data, results, time_indicator, time_period, amount_indicator, customer_indicator)
    top_100_data = data[data[customer_indicator].isin(top_100_customers)]
    worst_100_data = data[data[customer_indicator].isin(worst_100_customers)]

    filter_style = {
                    'borderBottom': 'thin lightgrey solid',
                    'backgroundColor': 'rgb(250, 250, 250)',
                    'padding': '10px 5px'
    }
    plot_ids = ['top_100_customer', 'worst_100_customer', 'time_line_of_values',
                'comparion_with_prediction', 'user_selection_comparison']
    plot_sizes = [99, 45, 45, 45, 45]
    hover_datas = get_hover_data(data, time_indicator, time_period, len(plot_ids))
    plot_dfs = []
    plots = list(zip(plot_ids, plot_sizes, hover_datas))
    # adding filters
    pane_count = int(len(filters) / num_f_p) if int(len(filters) / num_f_p) == len(filters) / num_f_p else int(
        len(filters) / num_f_p) + 1
    components = []
    for i in range(pane_count):
        _filters = filters[i * num_f_p:(i + 1) * num_f_p] if i != pane_count - 1 else filters[i * num_f_p:]
        _pane = adding_filter_to_pane([adding_filter(f[0], f[1], f[2], f[3], f[4]) for f in _filters], filter_style)
        components.append(_pane)
    # adding plots
    for p in plots:
        components.append(adding_plots_to_pane(p[0], p[2], p[1]))
    app.layout = html.Div(components)
    # top 100 customers
    @app.callback(
        dash.dependencies.Output(plot_ids[0], 'figure'),
        [dash.dependencies.Input(f, 'value') for f in [filter_ids[0], filter_ids[-1]]]
    )
    def update_graph_1(*args):
        f = filter_ids[0]
        dff = top_100_data[top_100_data[customer_indicator] == args[f]] if args[f] is not None else data
        value_column = amount_indicator + "_sum" if args[filter_ids[-1]] == 'sum' else amount_indicator + "_mean"
        func = {value_column: "sum"} if args[filter_ids[-1]] == 'sum' else {amount_indicator + "_mean": "mean"}
        dff = dff.groupby(time_indicator + '_per_' + time_period).agg(func).reset_index()
        title = "Top Customer " + args[f] if args[f] is not None else "Top 100 Customers"
        if len(dff) == 0:
            return {"data": [], "layout": go.Layout(height=600, title=title)}
        else:
            trace = go.Scatter(x=dff[time_indicator + '_per_' + time_period],
                               y=dff[value_column],
                               mode='markers+lines',
                               customdata=dff[time_indicator + '_per_' + time_period],
                               name=value_column)
            return {"data": trace,
                    "layout": go.Layout(height=600, title=title)}

    # worst 100 customers
    @app.callback(
        dash.dependencies.Output(plot_ids[1], 'figure'),
        [dash.dependencies.Input(f, 'value') for f in [filter_ids[1], filter_ids[-1]]]
    )
    def update_graph_2(*args):
        f = filter_ids[1]
        dff = worst_100_data[worst_100_data[customer_indicator] == args[f]] if args[f] is not None else data
        value_column = amount_indicator + "_sum" if args[filter_ids[-1]] == 'sum' else amount_indicator + "_mean"
        func = {value_column: "sum"} if args[filter_ids[-1]] == 'sum' else {amount_indicator + "_mean": "mean"}
        dff = dff.groupby(time_indicator + '_per_' + time_period).agg(func).reset_index()
        title = "Worst Customer " + args[f] if args[f] is not None else "Worst 100 Customers"
        if len(dff) == 0:
            return {"data": [], "layout": go.Layout(height=600, title=title)}
        else:
            trace = go.Scatter(x=dff[time_indicator + '_per_' + time_period],
                               y=dff[value_column],
                               mode='markers+lines',
                               customdata=dff[time_indicator + '_per_' + time_period],
                               name=value_column)
            return {"data": trace,
                    "layout": go.Layout(height=600, title=title)}

    # New Commers Of Customer Value Prediction
    """
    HoverData from clv time line selected date of customers who are newcomer of time line till the predicted time period
    """
    @app.callback(
        dash.dependencies.Output(plot_ids[2], 'figure'),
        [dash.dependencies.Input(f, 'value') for f in [filter_ids[-1]]] +
        [dash.dependencies.Input('daily-winners-line', 'hoverData')]
    )
    def update_graph_4(*args):
        value_column = amount_indicator + "_sum" if args[filter_ids[-1]][-1] == 'sum' else amount_indicator + "_mean"
        func = {value_column: "sum"} if args[filter_ids[-1]] == 'sum' else {amount_indicator + "_mean": "mean"}
        dff = data.groupby(time_indicator + '_per_' + time_period).agg(func).reset_index()
        title = "SUM of " if args[filter_ids[-1]] == 'sum' else "Average of "
        title += "CLV Prediction per " + time_period
        if len(dff) == 0:
            return {"data": [], "layout": go.Layout(height=600, title=title)}
        else:
            trace = go.Scatter(x=dff[time_indicator + '_per_' + time_period],
                               y=dff[value_column],
                               mode='markers+lines',
                               customdata=dff[time_indicator + '_per_' + time_period],
                               name=value_column)
            return {"data": trace,
                    "layout": go.Layout(height=600, title=title)}

    # Churn customers Of Customer Value Prediction
    """
    HoverData from clv time line selected date of customers who are newcomer of time line till the predicted time period
    """
    @app.callback(
        dash.dependencies.Output(plot_ids[2], 'figure'),
        [dash.dependencies.Input(f, 'value') for f in [filter_ids[-1]]] +
        [dash.dependencies.Input('daily-winners-line', 'hoverData')]
    )
    def update_graph_4(*args):
        starting_date = args['hoverData']['points'][0]['customdata']


        value_column = amount_indicator + "_sum" if args[filter_ids[-1]][-1] == 'sum' else amount_indicator + "_mean"
        func = {value_column: "sum"} if args[filter_ids[-1]] == 'sum' else {amount_indicator + "_mean": "mean"}
        dff = data.groupby(time_indicator + '_per_' + time_period).agg(func).reset_index()
        title = "SUM of " if args[filter_ids[-1]] == 'sum' else "Average of "
        title += "CLV Prediction per " + time_period
        if len(dff) == 0:
            return {"data": [], "layout": go.Layout(height=600, title=title)}
        else:
            trace = go.Scatter(x=dff[time_indicator + '_per_' + time_period],
                               y=dff[value_column],
                               mode='markers+lines',
                               customdata=dff[time_indicator + '_per_' + time_period],
                               name=value_column)
            return {"data": trace,
                    "layout": go.Layout(height=600, title=title)}


if __name__ == '__main__':
    create_dashboard()