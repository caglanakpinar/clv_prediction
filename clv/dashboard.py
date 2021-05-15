import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import datetime
import webbrowser
import argparse

try:
    from .data_access import GetData
    from .utils import convert_date
    from .functions import convert_time_preiod_to_days, get_results
except Exception as e:
    from data_access import GetData
    from utils import convert_date
    from functions import convert_time_preiod_to_days, get_results


def get_raw_data(time_indicator, amount_indicator, data_source, data_query_path):
    try:
        source = GetData(data_query_path=data_query_path,
                         data_source=data_source,
                         time_indicator=time_indicator,
                         feature=amount_indicator)
        source.query_data_source()
        source.convert_feature()
        data = source.data
    except Exception as e:
        data = pd.DataFrame()
        print("no data is available")
    return data


def get_filters(results, customer_indicator, amount_indicator):
    data_pv = results.groupby(customer_indicator).agg({amount_indicator: "sum"}).reset_index()
    top_100_customers = list(data_pv.sort_values(amount_indicator,ascending=False)[customer_indicator])[0:100]
    worst_100_customers = list(data_pv.sort_values(amount_indicator, ascending=True)[customer_indicator])[0:100]
    calculation_selection = ['sum', 'mean']
    filter_datas = [top_100_customers, worst_100_customers, calculation_selection]
    filter_ids = ['top_100_customers', 'worst_100_customers', 'calculation_type']
    filter_sizes = [100, 100, 20]
    multiple_selection = [False] * 3
    values = ['all', 'all', 'sum']
    filters = list(zip(filter_ids, filter_datas, filter_sizes, multiple_selection, values))
    return 3, filters, filter_ids, top_100_customers, worst_100_customers


def decide_time_period(date, time_period):
    if time_period == 'day':
        return datetime.datetime.strptime(str(date)[0:10], "%Y-%m-%d")
    if time_period in ['month', '2*month', 'quarter', '6*month']:
        return datetime.datetime.strptime(str(date)[0:7], "%Y-%m")
    if time_period == 'year':
        return datetime.datetime.strptime(str(date)[0:4], "%Y")
    if time_period == 'week':  # only shows mondays for weekly comparison
        week_day = datetime.datetime.strptime(str(date)[0:10], "%Y-%m-%d").isoweekday()
        return datetime.datetime.strptime(str(date)[0:10], "%Y-%m-%d") - datetime.timedelta(days=week_day-1)
    if time_period == '2*week':  # only shows mondays for weekly comparison
        date = datetime.datetime.strptime(str(date)[0:10], "%Y-%m-%d")
        week_day = date.isoweekday()
        week = date.isocalendar()[1]
        mondays_cal_day_count = week_day-1 if week % 2 == 0 else week_day - 1 + 7
        return datetime.datetime.strptime(str(date)[0:10], "%Y-%m-%d") - datetime.timedelta(days=mondays_cal_day_count)


def get_data_time_period_column(data, results, time_indicator, time_period):
    data['data_type'] = 'actual'
    results['data_type'] = 'prediction'
    data = pd.concat([data, results])
    data = data[data[time_indicator] == data[time_indicator]]
    data[time_indicator + '_per_' + time_period] = data[time_indicator].apply(
        lambda x: decide_time_period(x, time_period))
    return data


def get_new_comer_list(row, customer_list, time_p_ind, customer_indicator):
    prev_customers = list(customer_list[customer_list[time_p_ind] < row[time_p_ind]][customer_indicator])
    if len(prev_customers) != 0:
        prev_customers = np.concatenate(np.array(prev_customers)).tolist()
        diffs = list(set(row[customer_indicator]) - set(prev_customers))
        return pd.Series([diffs, len(diffs) / (len(prev_customers) + len(diffs))])
    else:
        return pd.Series([row[customer_indicator], 1])


def get_churn_list(row, customer_list, time_p_ind, customer_indicator, max_time_indicator):
    prediction_customers = list(customer_list[customer_list[time_p_ind] == max_time_indicator][customer_indicator])
    if len(row[customer_indicator]) != 0 and len(prediction_customers) != 0:
        prediction_customers = np.concatenate(np.array(prediction_customers)).tolist()
        diffs = list(set(row[customer_indicator]) - set(prediction_customers))
        return pd.Series([diffs, len(diffs) / len(set(prediction_customers +row[customer_indicator]))])
    else:
        return pd.Series([[], 0])


def get_churn_and_new_comer_columns(data, time_p_ind, amount_indicator, customer_indicator):
    data_pv = data.groupby(time_p_ind).agg(
        {amount_indicator + "_sum": "sum",
         amount_indicator + "_mean": "mean",
         customer_indicator: lambda x: list(np.unique(x))}).reset_index()

    data_pv = data_pv.sort_values(by=time_p_ind, ascending=True)
    customer_list = data_pv[[time_p_ind, customer_indicator]]
    predicted_time_period = max(customer_list[time_p_ind])
    data_pv[['new_comer_list', 'new_comer_ratio']] = data_pv.apply(
        lambda row: get_new_comer_list(row, customer_list, time_p_ind, customer_indicator), axis=1)
    data_pv[['churn_list', 'churn_ratio']] = data_pv.apply(
        lambda row: get_churn_list(row, customer_list, time_p_ind, customer_indicator, predicted_time_period), axis=1)
    data = pd.merge(data, data_pv[[time_p_ind, 'new_comer_list', 'new_comer_ratio']], on=time_p_ind, how='left')
    data = pd.merge(data, data_pv[[time_p_ind, 'churn_list', 'churn_ratio']], on=time_p_ind, how='left')
    return data, max(customer_list[time_p_ind])


def get_new_comer_churn_data(data, customer_indicator, amount_indicator, time_p_ind):
    data['is_new_comer'] = data.apply(lambda row: True if row[customer_indicator] in row['new_comer_list'] else False,
                                      axis=1)
    data['is_churn'] = data.apply(lambda row: True if row[customer_indicator] in row['churn_list'] else False, axis=1)
    new_comer_data = data.query("is_new_comer == True").groupby(time_p_ind).agg(
        {amount_indicator + "_sum": "sum",
         amount_indicator + "_mean": "mean", 'new_comer_ratio': 'first'}).reset_index()
    churn_data = data.query("is_churn == True").groupby(time_p_ind).agg(
        {amount_indicator + "_sum": "sum",
         amount_indicator + "_mean": "mean", 'churn_ratio': 'first'}).reset_index()
    return new_comer_data, churn_data


def pivoting_data_per_time_period(data,
                                  results,
                                  time_indicator,
                                  time_period,
                                  amount_indicator,
                                  customer_indicator,
                                  top_100_customers,
                                  worst_100_customers):
    time_p_ind = time_indicator + '_per_' + time_period
    data = get_data_time_period_column(data, results, time_indicator, time_period)
    data[amount_indicator + "_sum"], data[amount_indicator + "_mean"] = data[amount_indicator], data[amount_indicator]
    data = data[data[amount_indicator] == data[amount_indicator]]
    data_pv = data.groupby([time_p_ind, 'data_type']).agg(
        {amount_indicator + "_sum": "sum",
         amount_indicator + "_mean": "mean",
         customer_indicator: lambda x: list(np.unique(x))}).reset_index()
    data, predicted_time_period = get_churn_and_new_comer_columns(data, time_p_ind, amount_indicator, customer_indicator)
    new_comer_data, churn_data = get_new_comer_churn_data(data, customer_indicator, amount_indicator, time_p_ind)
    top_100_data = data[data[customer_indicator].isin(top_100_customers)]
    worst_100_data = data[data[customer_indicator].isin(worst_100_customers)]
    return data_pv.query("data_type == 'actual'"), data_pv.query("data_type == 'prediction'"), \
           top_100_data, worst_100_data, predicted_time_period, new_comer_data, churn_data


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
Time Line Chart with filter selected average/Total value per time period with predicted life time value
Top 100 customer of Time Period Change
Worst 100 Customer Of Time Period Value Change
New Comers of Time line with selected date from Time Line Chart
Churn Customers of Time line with selected date from Time Line Chart
"""


def get_hover_data(data, time_indicator, time_period, number_of_graph):
    value = sorted(list(data[time_indicator + '_per_' + time_period].unique()))
    return [{'customdata': str(value[-min(3, len(value))])}] * number_of_graph


def create_dashboard(customer_indicator, amount_indicator, directory,
                     time_indicator, time_period, data_query_path, data_source):
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
    try:
        data = get_raw_data(time_indicator, amount_indicator, data_source, data_query_path)
        results = get_results(directory, time_period)
        results = results[results[amount_indicator] == results[amount_indicator]]
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
    actual, prediction, top_100_data, worst_100_data, predicted_time_period, \
    new_comer_data, churn_data = pivoting_data_per_time_period(
                                                                data,
                                                                results,
                                                                time_indicator,
                                                                time_period,
                                                                amount_indicator,
                                                                customer_indicator,
                                                                top_100_customers,
                                                                worst_100_customers)

    filter_style = {
                    'borderBottom': 'thin lightgrey solid',
                    'backgroundColor': 'rgb(250, 250, 250)',
                    'padding': '10px 5px'
    }
    plot_ids = ['time_line_of_values', 'top_100_customer', 'worst_100_customer',
                'churn_customer', 'new_comer_customer', 'churn_rate', 'new_comer_engage_rate']
    plot_sizes = [99, 24, 24, 24, 24, 24, 24]
    hover_datas = get_hover_data(actual, time_indicator, time_period, len(plot_ids))
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
    # Churn Rate
    """
    HoverData from clv time line selected date of customers who are newcomer of time line till the predicted time period
    """
    @app.callback(
        dash.dependencies.Output(plot_ids[5], 'figure'),
        [dash.dependencies.Input('time_line_of_values', 'hoverData')]
    )
    def update_graph_4(hover_data):
        starting_date = hover_data['points'][0]['customdata']
        dff = churn_data[churn_data[time_indicator + '_per_' + time_period] == starting_date]
        if len(dff) == 0:
            return {"data": [], "layout": go.Layout(height=600, title="Churn Customers Of Purchase Time Line")}
        else:
            churn_rate = \
            list(churn_data[churn_data[time_indicator + '_per_' + time_period] == starting_date]['churn_ratio'])[0]
            return_rate = 1 - churn_rate
            trace = [go.Pie(labels=['Churn Rate', 'Return Rate'],
                            values=[churn_rate, return_rate],
                            )
                     ]
            return {"data": trace,
                    "layout": go.Layout(height=600, title=" Churn Rate (%) per " + time_period)}

    # New Comer Ratio
    """
    HoverData from clv time line selected date of customers who are newcomer of time line till the predicted time period
    """
    @app.callback(
        dash.dependencies.Output(plot_ids[6], 'figure'),
        [dash.dependencies.Input('time_line_of_values', 'hoverData')]
    )
    def update_graph_4(hover_data):
        starting_date = hover_data['points'][0]['customdata']
        dff = new_comer_data[new_comer_data[time_indicator + '_per_' + time_period] == starting_date]
        if len(dff) == 0:
            return {"data": [], "layout": go.Layout(height=600, title="Churn Customers Of Purchase Time Line")}
        else:
            newc_rate = \
            list(churn_data[churn_data[time_indicator + '_per_' + time_period] == starting_date]['churn_ratio'])[0]
            engaged_users = 1 - newc_rate
            trace = [go.Pie(labels=['Newcomer User Ratio', 'Engaged User Ratio'],
                            values=[newc_rate, engaged_users],
                            )
                     ]
            return {"data": trace,
                    "layout": go.Layout(height=600, title=" Newcomer Rate (%) per " + time_period)}

    # Churn Customers Of Purchase Time Line
    """
    HoverData from clv time line selected date of customers who are newcomer of time line till the predicted time period
    """
    @app.callback(
        dash.dependencies.Output(plot_ids[3], 'figure'),
        [dash.dependencies.Input(f, 'value') for f in [filter_ids[-1]]] +
        [dash.dependencies.Input('time_line_of_values', 'hoverData')]
    )
    def update_graph_4(sum_or_avg, hover_data):
        starting_date = hover_data['points'][0]['customdata']
        value_column = amount_indicator + "_sum" if sum_or_avg == 'sum' else amount_indicator + "_mean"
        dff = churn_data[churn_data[time_indicator + '_per_' + time_period] >= starting_date]
        func = {value_column: "sum"} if sum_or_avg == 'sum' else {amount_indicator + "_mean": "mean"}
        dff = dff.groupby(time_indicator + '_per_' + time_period).agg(func).reset_index()
        if len(dff) == 0:
            return {"data": [], "layout": go.Layout(height=600, title="Churn Customers Of Purchase Time Line")}
        else:
            trace = [go.Scatter(x=dff[time_indicator + '_per_' + time_period],
                                y=dff[value_column],
                                mode='markers+lines',
                                name=value_column)
                     ]
            return {"data": trace,
                    "layout": go.Layout(height=600, title=" Churn Customers Of Purchase Time Line")}

    # top 100 customers
    @app.callback(
        dash.dependencies.Output(plot_ids[1], 'figure'),
        [dash.dependencies.Input(f, 'value') for f in [filter_ids[0], filter_ids[-1]]]
    )
    def update_graph_1(_customer, sum_or_avg):
        dff = top_100_data[top_100_data[customer_indicator] == _customer] if _customer != 'all' else top_100_data
        value_column = amount_indicator + "_sum" if sum_or_avg == 'sum' else amount_indicator + "_mean"
        func = {value_column: "sum"} if sum_or_avg == 'sum' else {amount_indicator + "_mean": "mean"}
        dff = dff.groupby(time_indicator + '_per_' + time_period).agg(func).reset_index()
        title = "Top 100 the most Engaged Customers of Average Values per " + time_period
        if sum_or_avg == 'sum':
            title = title.replace("Average", "Sum")
        if len(dff) == 0:
            return {"data": [], "layout": go.Layout(height=600, title=title)}
        else:
            trace = go.Scatter(x=dff[time_indicator + '_per_' + time_period],
                               y=dff[value_column],
                               mode='markers+lines',
                               name=value_column)
            return {"data": [trace],
                    "layout": go.Layout(height=600, title=title)}

    # worst 100 customers
    @app.callback(
        dash.dependencies.Output(plot_ids[2], 'figure'),
        [dash.dependencies.Input(f, 'value') for f in [filter_ids[1], filter_ids[-1]]]
    )
    def update_graph_2(_customer, sum_or_avg):
        dff = worst_100_data[worst_100_data[customer_indicator] == _customer] if _customer != 'all' else worst_100_data
        value_column = amount_indicator + "_sum" if sum_or_avg == 'sum' else amount_indicator + "_mean"
        func = {value_column: "sum"} if sum_or_avg == 'sum' else {amount_indicator + "_mean": "mean"}
        dff = dff.groupby(time_indicator + '_per_' + time_period).agg(func).reset_index()
        title = "Top 100 the least Engaged Customers of Average Values per " + time_period
        if sum_or_avg == 'sum':
            title = title.replace("Average", "Sum")
        if len(dff) == 0:
            return {"data": [], "layout": go.Layout(height=600, title=title)}
        else:
            trace = [go.Scatter(x=dff[time_indicator + '_per_' + time_period],
                                y=dff[value_column],
                                mode='markers+lines',
                                name=value_column)]
            return {"data": trace,
                    "layout": go.Layout(height=600, title=title)}

    # Time Line Of CLV Prediction
    """
    HoverData from clv time line selected date of customers who are newcomer of time line till the predicted time period
    """
    @app.callback(
        dash.dependencies.Output(plot_ids[0], 'figure'),
        [dash.dependencies.Input(f, 'value') for f in [filter_ids[-1]]]
    )
    def update_graph_4(sum_or_avg):
        value_column = amount_indicator + "_sum" if sum_or_avg == 'sum' else amount_indicator + "_mean"
        title = "SUM of " if sum_or_avg == 'sum' else "Average of "
        title += "CLV Prediction per " + time_period
        _prev_time_period = predicted_time_period - datetime.timedelta(days=convert_time_preiod_to_days(time_period))

        if len(actual) == 0 and len(prediction) == 0:
            return {"data": [], "layout": go.Layout(height=600, title=title)}
        else:
            trace = [go.Scatter(x=actual[time_indicator + '_per_' + time_period],
                                y=actual[value_column],
                                mode='markers+lines',
                                customdata=actual[time_indicator + '_per_' + time_period],
                                name=value_column + " Actual"),
                     go.Scatter(x=prediction[time_indicator + '_per_' + time_period],
                                y=prediction[value_column],
                                mode='markers+lines',
                                name='CLV Prediction')]
            return {"data": trace,
                    "layout": go.Layout(height=600, title=title)}

    # Newcomer Customers Of Customer Value Prediction
    """
    HoverData from clv time line selected date of customers who are newcomer of time line till the predicted time period
    """
    @app.callback(
        dash.dependencies.Output(plot_ids[4], 'figure'),
        [dash.dependencies.Input(f, 'value') for f in [filter_ids[-1]]] +
        [dash.dependencies.Input('time_line_of_values', 'hoverData')]
    )
    def update_graph_4(sum_or_avg, hover_data):
        starting_date = hover_data['points'][0]['customdata'] # '2018-01-01 00:00:00'
        value_column = amount_indicator + "_sum" if sum_or_avg == 'sum' else amount_indicator + "_mean"
        dff = new_comer_data[new_comer_data[time_indicator + '_per_' + time_period] >= starting_date]
        if len(dff) == 0:
            return {"data": [], "layout": go.Layout(height=600,
                                                    title="Newcomer Customer Value Prediction")}
        else:
            trace = [go.Scatter(x=dff[time_indicator + '_per_' + time_period],
                                y=dff[value_column],
                                mode='markers+lines',
                                customdata=dff[time_indicator + '_per_' + time_period],
                                name=value_column)]
            return {"data": trace,
                    "layout": go.Layout(height=600, title="Newcomer Customer Value Prediction")}

    webbrowser.open('http://127.0.0.1:8050/')
    app.run_server(debug=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-CI", "--customer_indicator", type=str,
                        help="""identifier of the customer (id)
##
                        """,
                        )
    parser.add_argument("-AI", "--amount_indicator", type=str,
                        help="""
                                data column n-amount (preferred numeric)
                        """,
                        )
    parser.add_argument("-TI", "--time_indicator", type=str,
                        help="""
                        This can only be applied with date. It can be only day..
                        Individually time indicator checks the date part is significantly
                        a individual group for data set or not.
                        If it is uses time_indicator as a  group
                        """,
                        )
    parser.add_argument("-TP", "--time_period", type=str,
                        help="""
                        This shows us to the time period
                        """,
                        )
    parser.add_argument("-EP", "--export_path", type=str,
                        help="""
                        Exporting path of the results set. Csv file of exporting.
                        """,
                        )
    parser.add_argument("-DS", "--data_source", type=str,
                        help="""
                        AWS RedShift, BigQuery, PostgreSQL, csv, json files can be connected to system
                        """,
                        required=True)
    parser.add_argument("-DQP", "--data_query_path", type=str,
                        help="""
                        if there is file for data importing;
                            must be the path (e.g /../.../ab_test_raw_data.csv)
                        if there is ac- connection such as PostgreSQL / BigQuery
                            query must at the format "SELECT++++*+++FROM++ab_test_table_+++"
                        """,
                        required=True)
    arguments = parser.parse_args()
    args = {
            'customer_indicator': arguments.customer_indicator,
            'amount_indicator': arguments.amount_indicator,
            'directory': arguments.export_path,
            'time_period': arguments.time_period,
            'time_indicator': arguments.time_indicator,
            'data_source': arguments.time_period,
            'data_query_path': arguments.data_query_path}
    print(args)
    create_dashboard(**args)