import dash_bootstrap_components as dbc
import dash

from dash import dcc, html, Input, Output, State, ctx, ALL, callback
import components as c
from typing import List
from probabilistic_model.probabilistic_circuit.nx.probabilistic_circuit import ProbabilisticCircuit
import random_events.variable

global result
global page
page = 0

global modal_var_index
modal_var_index = -1

global modal_basic_pos
modal_basic_pos = c.gen_modal_basic_id("_pos")

modal_option_pos = c.gen_modal_option_id("_pos")

dash.register_page(__name__)


def layout_pos():
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(html.H1("Posterior", className='text-center mb-4'), width=12),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col([
                        html.Div("P ", className="ps-3",
                                 style={'fontSize': 30, 'padding-top': 0}),
                    ], id="text_l_pos", align="center",
                        className="d-flex flex-wrap align-items-center justify-content-end pe-3", width=2),
                    dbc.Col(id="q_variable_pos",
                            children=[
                                dcc.Dropdown(id="text_var_pos", options=sorted(c.vardict.keys()),
                                             value=sorted(c.vardict.keys()),
                                             multi=True, disabled=False)],
                            width=4, className="row row-cols-1 g-1 gy-2 align-items-center border-start border-3 rounded-4 border-secondary"),
                    dbc.Col(id="e_variable_pos",
                            children=[dcc.Dropdown(id={'type': 'dd_e_pos', 'index': 0},
                                                   options=sorted(c.vardict.keys()))],
                            width=2, className="row row-cols-1 g-1 gy-2 align-items-center border-start border-3 border-secondary"),
                    dbc.Col(id="e_input_pos",
                            children=[dcc.Dropdown(id={'type': 'i_e_pos', 'index': 0}, disabled=True)], width=3,
                            className="row row-cols-1 g-1 gy-2 align-items-center"),
                    dbc.Col(id="e_option_pos", children=[
                        dbc.Button("👁️", id=dict(type='b_e_pos', index=0), disabled=True, n_clicks=0,
                                   className="",
                                   size="sm", style={'width': '40px'})],className="row row-cols-1 g-1 gy-2 align-items-center pe-3 ps-1 border-end border-secondary border-3 rounded-4"),
                    dbc.Col()
                ], className="row row-cols-8 g-1 gy-2 mb-3"
            ),
            dbc.Row(dbc.Button("=", id="erg_b_pos", className="d-grid gap-2 col-3 mt-3 mx-auto", n_clicks=0)),
            dbc.Row(dbc.Col(html.H2("", className='text-center mb-4', id="head_erg_pos"), className="pt-3", width=12)),
            dbc.Row(
                [
                    dbc.Col(dbc.Button("<", id="b_erg_pre_pos", n_clicks=0, disabled=True),
                            className="d-flex justify-content-end align-self-stretch"),
                    dbc.Col(children=[], id="pos_erg_pos", className="", width=8),
                    dbc.Col(dbc.Button(">", id="b_erg_next_pos", n_clicks=0, disabled=True),
                            className="d-flex justify-content-start align-self-stretch")
                ], className="pt-3", align="center"),
            dbc.Row(),
            modal_option_pos
        ], fluid=True
    )


layout = layout_pos


@callback(
    Output('e_variable_pos', 'children'),
    Output('e_input_pos', 'children'),
    Output('e_option_pos', 'children'),
    Output('text_l_pos', 'children'),
    Output('q_variable_pos', 'children'),
    Output('modal_option_pos', 'children'),
    Output('modal_option_pos', 'is_open'),
    Input({'type': 'dd_e_pos', 'index': ALL}, 'value'),
    Input({'type': 'b_e_pos', 'index': ALL}, 'n_clicks'),
    Input({'type': 'option_save_pos', 'index': ALL}, 'n_clicks'),
    State('e_variable_pos', 'children'),
    State('e_input_pos', 'children'),
    State('q_variable_pos', 'children'),
    State('e_option_pos', 'children'),
    State({'type': 'op_i_pos', 'index': ALL}, 'value'),
)
def post_router(dd_vals, b_e, op_s, e_var, e_in, q_var, e_op, op_i):
    """
        Receives callback events and manages these to the correct
    :param dd_vals: All Varietals used in Evidence Section are chosen
    :param b_e: Trigger if the Zoom Button in the Evidence is Pressed
    :param op_s: Trigger if the Modal parameter from a Zoom should be saved
    :param e_var: the Dropdown of variable of Evidence Section
    :param e_in: the Input for the Variables of Evidence Section
    :param q_var: the Dropdown of variable of Query Section
    :param e_op: Information of whiche Zoom Button was pressed in the Evidence section
    :param op_i: The Values choosen in the Zoom Modal
    :return: returns evidence variable, evidence Input, text prefix, query Variable
    """
    cb = ctx.triggered_id if not None else None
    if cb is None:
        return e_var, e_in, e_op, c.create_prefix_text_query(len(e_var), len(e_var)), q_var, modal_basic_pos, False
    elif cb.get("type") == "dd_e_pos":
        if dd_vals[cb.get("index")] is None:
            return *c.del_selector_from_div_button(c.in_use_model, e_var, e_in, e_op, cb.get("index")), \
                c.create_prefix_text_query(len(e_var), len(e_var)), q_var, modal_basic_pos, False

        variable = c.vardict[dd_vals[cb.get("index")]]
        e_in[cb.get("index")] = c.correct_input_div(variable=variable, id={'type': 'i_e_pos', 'index': cb.get("index")}, priors=c.prior)

        if len(e_var) - 1 == cb.get("index"):
            return *c.add_selector_to_div_button(c.in_use_model, e_var, e_in, e_op, "e_pos", cb.get("index") + 1), \
                c.create_prefix_text_query(len(e_var), len(e_var)), q_var, modal_basic_pos, False
    elif cb.get("type") == "b_e_pos" and dd_vals[cb.get("index")] != []:
        # Dont Like dont know to do it other wise
        global modal_var_index
        modal_var_index = cb.get("index")
        variable = c.vardict[dd_vals[cb.get("index")]]
        modal_body = List
        if isinstance(variable, random_events.variable.Continuous):
            modal_body = c.generate_modal_option(model=c.in_use_model, var=e_var[cb.get("index")]['props']['value'],
                                                 value=[e_in[cb.get("index")]['props']['min'],
                                                        e_in[cb.get("index")]['props']['max']],
                                                 priors=c.prior, id="_pos")
        elif isinstance(variable, random_events.variable.Symbolic):
            modal_body = c.generate_modal_option(model=c.in_use_model, var=e_var[cb.get("index")]['props']['value'],
                                                 value=e_in[cb.get("index")]['props'].get('value'), priors=c.prior,
                                                 id="_pos")
        elif isinstance(variable, random_events.variable.Integer):
            modal_body = c.generate_modal_option(model=c.in_use_model, var=e_var[cb.get("index")]['props']['value'],
                                                 value=[e_in[cb.get("index")]['props']['min'],
                                                        e_in[cb.get("index")]['props']['max']],
                                                 priors=c.prior, id="_pos")

        return e_var, e_in, e_op, c.create_prefix_text_query(len(e_var), len(e_var)), q_var, modal_body, True
    elif cb.get("type") == "option_save_pos":
        variable = c.vardict[dd_vals[cb.get("index")]]
        new_vals = List
        if isinstance(variable, random_events.variable.Continuous) or isinstance(variable, random_events.variable.Integer):
            new_vals = c.fuse_overlapping_range(op_i)
        else:
            new_vals = op_i[0]#is List of a List
        e_in[modal_var_index]['props']['value'] = new_vals
        e_in[modal_var_index]['props']['drag_value'] = new_vals
        return e_var, e_in, e_op, c.create_prefix_text_query(len(e_var), len(e_var)), q_var, modal_basic_pos, False

    return c.update_free_vars_in_div(c.in_use_model, e_var), e_in, e_op, c.create_prefix_text_query(len(e_var),
                                                                                                   len(e_var)), \
        q_var, modal_basic_pos, False


@callback(
    Output("modal_input_pos", "children"),
    Input("op_add_pos", "n_clicks"),
    Input({'type': 'op_i_pos', 'index': ALL}, 'value'),
    State("modal_input_pos", "children"),
    State({'type': 'dd_e_pos', 'index': ALL}, 'value'),
)
def modal_router(op, op_i, m_bod, dd):
    """
        Recessive all App Calls that are change the Modal for the zoom Function
    :param op: Trigger to add More Input Option by Numeric Variabel
    :param op_i: Trigger to update Chance for the Chosen values
    :param m_bod: The State of the Modal
    :param dd: div withe the chosen values
    :return: update Modal Body for the Zoom
    """
    cb = ctx.triggered_id if not None else None
    if cb is None:
        return m_bod
    global modal_var_index
    var = dd[modal_var_index]

    if not isinstance(m_bod, list):
        m_in_new = [m_bod]
    else:
        m_in_new = m_bod
    if cb == "op_add_pos":
        index = m_in_new[-2]['props']['children'][0]['props']['children'][1]['props']['id']['index']
        type = m_in_new[1]['props']['children'][0]['props']['children'][1]['type']
        return c.modal_add_input(body=m_in_new, id_type='op_i_pos', index=index,  var=var)
        # if isinstance(variable, random_events.variable.Continuous):
        #
        #     mini = m_in_new[1]['props']['children'][0]['props']['children'][1]['props']['min']
        #     maxi = m_in_new[1]['props']['children'][0]['props']['children'][1]['props']['max']
        #     range_string = html.Div(f"Range {index + 2}",
        #                             style=dict(color=c.color_list_modal[(index + 1) % (len(c.color_list_modal) - 1)]))
        #     n_slider = c.create_range_slider(minimum=mini, maximum=maxi, id={'type': 'op_i_pos', 'index': index + 1},
        #                                      value=[mini, maxi], dots=False,
        #                                      tooltip={"placement": "bottom", "always_visible": False},
        #                                      className="flex-fill")
        #     var_event = c.div_to_event(c.in_use_model, [var], [[mini, maxi]])
        #     prob = c.in_use_model.probability(var_event.as_composite_set())
        #
        #     prob_div = html.Div(f"{round(prob, 5)}",
        #                         style=dict(color=c.color_list_modal[(index + 1) % (len(c.color_list_modal) - 1)]))
        #     m_in_new.insert(len(m_in_new) - 1, dbc.Row([
        #         html.Div([range_string, n_slider, prob_div],
        #                  id=f"modal_color_{(index + 1) % (len(c.color_list_modal) - 1)}",
        #                  className="d-flex flex-nowrap justify-content-center ps-2")
        #     ], className="d-flex justify-content-center"))
        #     return m_in_new
        # elif isinstance(variable, random_events.variable.Integer):
        #     lab = list(variable.domain[variable])
        #     mini = min(lab)
        #     maxi = max(lab)
        #     markings = dict(zip(lab, map(str, lab)))
        #     range_string = html.Div(f"Range {index + 2}",
        #                             style=dict(color=c.color_list_modal[(index + 1) % (len(c.color_list_modal)-1)]))
        #     n_slider = c.create_range_slider(minimum=mini, maximum=maxi, value=[mini, maxi]
        #                                                   ,id={'type': 'op_i_pos', 'index': index + 1}, dots=False,
        #                                                   marks=markings,
        #                                                   tooltip={"placement": "bottom", "always_visible": False},
        #                                     className="flex-fill")
        #     var_event = c.div_to_event(c.in_use_model, [var], [[mini, maxi]])
        #     prob = c.in_use_model.probability(var_event.as_composite_set())
        #     prob_div = html.Div(f"{round(prob, 5)}", style=dict(color=c.color_list_modal[(index + 1) % (len(c.color_list_modal)-1)]))
        #     m_in_new.insert(len(m_in_new) - 1, dbc.Row([
        #         html.Div([range_string, n_slider, prob_div], id=f"modal_color_{(index + 1) % (len(c.color_list_modal)-1)}", className="d-flex flex-nowrap justify-content-center ps-2")
        #     ],className="d-flex justify-content-center"))
        #     return m_in_new
        # else:
        #     # Sollte nicht Triggerbar sein, da bei DDMenu der +Buttone nicht Aktiv ist
        #     return m_in_new
    else:  # if cb.get("type") == "op_i"
        index = cb.get("index")
        return c.modal_save_input(body=m_in_new, index=index, var=var)
        # value = m_in_new[id + 1]['props']['children'][0]['props']['children'][1]['props']['value']
        # var_event = c.div_to_event(c.in_use_model, [var], [value])
        # prob = c.in_use_model.probability(var_event.as_composite_set())
        # prob_div = html.Div(f"{round(prob, 5)}", style=dict(color=c.color_list_modal[id % (len(c.color_list_modal) - 1)]))
        # m_in_new[id + 1]['props']['children'][0]['props']['children'][2] = prob_div
        # return m_in_new


@callback(
    Output('head_erg_pos', 'children'),
    Output('pos_erg_pos', 'children'),
    Output('b_erg_pre_pos', 'disabled'),
    Output('b_erg_next_pos', 'disabled'),
    Input('erg_b_pos', 'n_clicks'),
    Input('b_erg_pre_pos', 'n_clicks'),
    Input('b_erg_next_pos', 'n_clicks'),
    State({'type': 'dd_e_pos', 'index': ALL}, 'value'),
    State({'type': 'i_e_pos', 'index': ALL}, 'value'),
    State('q_variable_pos', 'children'),
)
def erg_controller(n1, n2, n3, e_var, e_in, q_var):
    """
        Conntroller for the Results and the Displays
    :param n1: event for generating Result
    :param n2: the Previous Result
    :param n3: the Next Result
    :param e_var: the Dropdown of variable of Evidence Section
    :param e_in: the Input for the Variables of Evidence Section
    :param q_var: the Dropdown of variable of Query Section
    :return: Returns the Name of The Variabel, the plot of the Variable, if there is a pre or post result
    """
    global result
    global page
    vals = q_var[0]['props']['value']
    cb = ctx.triggered_id if not None else None
    if cb is None:
        return [], [], True, True
    if cb == "b_erg_pre_pos":
        page -= 1
        if page == 0:
            return vals[page], plot_post(vals, page, result), True, False
        else:
            return vals[page], plot_post(vals, page, result), False, False
    elif cb == "b_erg_next_pos":
        page += 1
        if len(vals) > page + 1:
            return vals[page], plot_post(vals, page, result), False, False
        else:
            return vals[page], plot_post(vals, page, result), False, True
    elif vals == [] or cb == "b_erg_pos":
        return [], [], True, True
    else:
        page = 0
        evidence_dict = c.div_to_event(c.in_use_model, e_var, e_in)
        try:
            result = c.calculate_posterior_distributions(evidence_dict, c.in_use_model)
        except Exception as e:
            print("Error was", type(e), e)
            return "", [html.Div("Unsatisfiable", className="fs-1 text text-center pt-3 ")], True, True
        if len(vals) > 1:
            return vals[page], plot_post(vals, page, result), True, False
        else:
            return vals[page], plot_post(vals, page, result), True, True


def plot_post(vars: List, n: int, result):
    """
        Generates the Plots for a Varibel in Vars postion n
    :param vars: List of Variabel
    :param n: Postion of the Choosen Variabel
    :return:  Plot
    """
    var_name = vars[n]
    variable = c.vardict[var_name]
    return c.generate_plot_for_variable(variable, result)