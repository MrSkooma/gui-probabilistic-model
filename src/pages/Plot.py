import dash_bootstrap_components as dbc
import dash

from dash import dcc, html, Input, Output, State, ctx, ALL, callback
import components as c
from typing import List
from probabilistic_model.probabilistic_circuit.probabilistic_circuit import ProbabilisticCircuit
import random_events.variable


dash.register_page(__name__)

def layout_plot():
    return dbc.Container([
        dbc.Row(
            [
                dbc.Col(html.H1("Plot"))
            ]
        ),
        dbc.Row(
            [
                dcc.Graph(id="graph", figure=c.plot_3d(c.in_use_model), config={"displayModeBar": False})
            ]
        )
    ]

    )

layout = layout_plot