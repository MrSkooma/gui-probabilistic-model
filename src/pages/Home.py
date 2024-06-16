import dash
import random_events.variable
from dash import html, Input, Output, callback
import components as c
import dash_bootstrap_components as dbc
from random_events.set import  Set
from random_events.interval import Interval, SimpleInterval
from probabilistic_model.probabilistic_circuit.probabilistic_circuit import ProbabilisticCircuit
from random_events.product_algebra import Event
dash.register_page(__name__, path='/')


@callback(
    Output('list', 'children'),
    Input('list', 'children')
)
def gen_varnames(children):
    var_divs = []
    if c.vardict is None or (len(c.vardict) <= 1):
        return var_divs
    for variable, distrubiton in c.prior.items():
        var_name = variable.name
        dis_val_event = c.prior[variable].support().simple_sets[0][variable]
        if isinstance(variable, random_events.variable.Continuous) or isinstance(variable, random_events.variable.Integer):
            dis_val_event: Interval
            mini = dis_val_event.simple_sets[0].lower
            maxi = dis_val_event.simple_sets[-1].upper
            childStr = [html.Div(var_name, className="fs-4  flex-nowrap flex-grow-0 text-nowrap text-start"), html.Div(" ∈ ", className="pe-2 ps-1 fs-4  flex-nowrap flex-grow-0 text-nowrap text-start"), html.Div(f"[{round(mini,3)}, {round(maxi, 3)}]", className="fs-4  flex-nowrap flex-grow-0 text-nowrap text-start")]
            var_divs.append(html.Div(childStr, className="d-flex justify-content-center flex-grow-0"))
        else:
            dis_val_event: Set
            vals = [v.name for v in dis_val_event.simple_sets]
            childStr = [html.Div(var_name, className="fs-4 flex-nowrap flex-grow-0 text-nowrap text-start"), html.Div(" ∈ ", className="pe-2 ps-1 fs-4 flex-nowrap flex-grow-0 text-nowrap text-start"), html.Div(f"({vals})", className="fs-4 flex-nowrap flex-grow-0 text-nowrap text-start")]
            var_divs.append(html.Div(childStr, className="d-flex justify-content-center flex-grow-0"))
    return var_divs

    return html.Div(children=var_divs)


layout = html.Div([
    dbc.Row(html.H1("Home"), className="d-flex justify-content-center"),
    dbc.Row(dbc.Col(html.Img(src="./assets/Logo.svg", height="350px"), width=3), className="d-flex justify-content-center mb-3"),
    dbc.Row(html.Div(children=[], id="list", className=""), className="d-flex justify-content-center"),
])

# Home Name , Type , Range, AUGEN EMOTION HTML Sektion bigger and better