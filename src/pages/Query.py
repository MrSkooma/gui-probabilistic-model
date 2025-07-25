
import dash_bootstrap_components as dbc
import dash
import random_events.variable
from dash import dcc, html, Input, Output, State, ctx, ALL, callback
import components as c
from typing import List
from probabilistic_model.probabilistic_circuit.nx.probabilistic_circuit import ProbabilisticCircuit

import random_events
import random_events.product_algebra as pa
"""
On this Page there can be ask Query Situation. 
Query: What the Propablity is for the Attributes to be in the Range/Style 
Evidence: What is Given for the Probablity of the Query 
After = Button there will be the Display of the Percantage 
"""

global modal_var_index
modal_var_index = -1

global modal_type
# 0 = q and 1 = e
modal_type = -1

global modal_basic_que
modal_basic_que = c.gen_modal_basic_id("_que")

modal_option_que = c.gen_modal_option_id("_que")

# global old_time
# old_time = 0
#
# global time_list
# time_list = [dict()]

dash.register_page(__name__)


def layout_que():
    """
        Generates the Basic Layout in Dash for Query withe the Tree varnames as Options
    :return: Dash html strucktur
    """
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(html.H1("Query", className='text-center mb-4'), width=12),
                ]
            ),
            # dbc.Row([
            #     dbc.Col([dbc.Button("-", n_clicks=0, className="align-self-start mt-0 flex-grow-0", size="sm",
            #                         id="que_timer_minus", style={'verticalAlign': 'top', 'width': '40px'})], width=1,
            #             className="ps-0 pe-0 mb-2 mt-0 row row-cols-1 g-1 gy-2 justify-content-end"),
            #     dbc.Col([dcc.RangeSlider(min=1, max=1, value=[1], step=1, dots=False,
            #                              tooltip={"placement": "bottom", "always_visible": False}, id="que_time")],
            #             width=10, className="pe-0 pt-3"),
            #     dbc.Col(children=[
            #         dbc.Button("+", n_clicks=0, className="mt-0 flex-grow-0 align-self-start", size="sm",
            #                    id="que_timer_plus", style={'verticalAlign': 'top', 'width': '40px'})],
            #         width=1, className="ps-0 pe-0 mb-2 mt-0 row row-cols-1 g-1 gy-2 justify-content-start"),
            # ], className="mt-2 d-flex justify-content-center align-items-center"
            # ),

            dbc.Row(
                [
                    dbc.Col([
                        html.Div("P ", className="",  # align-self-center text-end float-end
                                 style={'fontSize': 40, 'padding-top': 0, 'padding-left': 0}),
                    ], id="text_l_que", align="center", className="", width=1),
                    dbc.Col(id="q_variable_que",
                            children=[
                                dcc.Dropdown(id={'type': 'dd_q_que', 'index': 0}, options=sorted(c.vardict.keys()),
                                             className="")],
                            width=2,
                            className="row row-cols-1 g-1 gy-2 align-items-center border-start border-secondary border-3 rounded-4"),
                    # d-grid gap-3 border-start border-secondary border-3 rounded-4
                    dbc.Col(id="q_input_que",
                            children=[dcc.Dropdown(id={'type': 'i_q_que', 'index': 0}, disabled=True)], width=3,
                            className="row row-cols-1 g-1 gy-2 align-items-center"),  # d-grid gap-3
                    dbc.Col(id="q_option_que", children=[
                        dbc.Button("👁️", id=dict(type='b_q_que', index=0), disabled=True, n_clicks=0, className="",
                                   size="sm")],
                            className="row row-cols-1 g-1 gy-2 align-items-center pe-3 ps-1 border-end border-3 border-secondary")
                    # d-grid align-self-start
                    ,
                    # d-grid gap-0 gx-0 d-flex align-items-stretch flex-grow-0 align-self-stretch border-end border-3 border-secondary
                    dbc.Col(id="e_variable_que",
                            children=[
                                dcc.Dropdown(id={'type': 'dd_e_que', 'index': 0}, options=sorted(c.vardict.keys()))],
                            width=2,
                            className="row row-cols-1 g-1 gy-2 align-items-center border-start border-3 border-secondary"),
                    # d-grid gap-0 border-start border-3 border-secondary ps-3
                    dbc.Col(id="e_input_que",
                            children=[dcc.Dropdown(id={'type': 'i_e_que', 'index': 0}, disabled=True)], width=3,
                            className="row row-cols-1 g-1 gy-2 align-items-center"),  # d-grid gap-3
                    dbc.Col(id="e_option_que", children=[
                        dbc.Button("👁️", id=dict(type='b_e_que', index=0), disabled=True, n_clicks=0, className="",
                                   size="sm")],
                            className="row row-cols-1 g-1 gy-2 align-items-center pe-3 ps-1 border-end border-secondary border-3 rounded-4"),
                    # d-grid border-end border-secondary border-3 rounded-4 #d-grid gx-1 d-md-flex align-self-center
                ], className="row row-cols-8 g-1 gy-2 mb-3"  # justify-content-center
            ),
            dbc.Row(dbc.Button("=", id="erg_b_que", className="d-grid gap-2 col-6 mx-auto", n_clicks=0)),
            # d-grid gap-2 col-3 mt-3 mx-auto
            dbc.Row(dbc.Col(html.Div("", id="erg_text_que", className="fs-1 text text-center pt-3"))),
            modal_option_que
        ], fluid=True
    )


layout = layout_que


def query_gen(dd_vals: List, q_var: List, q_in: List, q_op):
    """
        Handel all action in the Query Part of the GUI (Extend Change Reduce)
    :param dd_vals: All Varietals used in Query Section are chosen
    :param q_var: the Dropdown of variable of Query Section
    :param q_in: the Input for the Variables of Query Section
    :param q_op:  the Variabel  who is selected for the Zoom
    :return: Updatet Varibel List and the Input.
    """
    cb = ctx.triggered_id
    if dd_vals[cb.get("index")] is None:
        return c.del_selector_from_div_button(c.in_use_model, q_var, q_in, q_op, cb.get("index"))
    variable = c.vardict[dd_vals[cb.get("index")]]
    q_in[cb.get("index")] = c.correct_input_div(variable=variable, id={'type': 'i_q_que', 'index': cb.get("index")}, priors=c.prior)
    # if isinstance(variable, random_events.variable.Continuous):
    #     minimum = c.prior[variable].domain.events[0][variable].lower
    #     maximum = c.prior[variable].domain.events[0][variable].upper
    #
    #      c.create_range_slider(minimum, maximum,
    #                                                   id={'type': 'i_q_que', 'index': cb.get("index")},
    #                                                   tooltip={"placement": "bottom", "always_visible": False},
    #                                                   className="")
    #
    # elif isinstance(variable, random_events.variable.Symbolic):
    #     q_in[cb.get("index")] = dcc.Dropdown(id={"type": "i_q_que", "index": cb.get("index")},
    #                                          options={k: v for k, v in zip(variable.domain,
    #                                                                        variable.domain)},
    #                                          value=list(variable.domain),
    #                                          multi=True, )  # list(variable.domain.labels.keys())
    # elif isinstance(variable, random_events.variable.Integer):
    #     lab = list(variable.domain)
    #     mini = min(lab)
    #     maxi = max(lab)
    #     markings = dict(zip(lab, map(str, lab)))
    #     q_in[cb.get("index")] = c.create_range_slider(minimum=mini - 1, maximum=maxi + 1, value=[mini, maxi],
    #                                                   id={'type': 'i_q_que', 'index': cb.get("index")}, dots=False,
    #                                                   marks=markings,
    #                                                   tooltip={"placement": "bottom", "always_visible": False})

    if len(q_var) - 1 == cb.get("index"):
        return c.add_selector_to_div_button(c.in_use_model, q_var, q_in, q_op, 'q_que', cb.get("index") + 1)
    return c.update_free_vars_in_div(c.in_use_model, q_var), q_in, q_op


def evid_gen(dd_vals, e_var, e_in, e_op):
    """
        Handel all action in the Evidence Part of the GUI (Extend Change Reduce)
    :param dd_vals: All Varietals used in Evidence Section are chosen
    :param e_var: the Dropdown of variable of Evidence Section
    :param e_in: the Input for the Variables of Evidence Section
    :param q_op:  the Variabel  who is selected for the Zoom
    :return: Updatet Varibel List and the Input.
    """
    e_var: List[dict] = e_var
    e_in: List[dict] = e_in
    cb = ctx.triggered_id
    if dd_vals[cb.get("index")] is None:
        return c.del_selector_from_div_button(c.in_use_model, e_var, e_in, e_op, cb.get('index'))

    variable = c.vardict[dd_vals[cb.get("index")]]
    e_in[cb.get("index")] = c.correct_input_div(variable=variable, id={'type': 'i_e_que', 'index': cb.get("index")}, priors=c.prior)
    # if isinstance(variable, random_events.variable.Continuous):
    #     minimum = c.prior[variable].domain.events[0][variable].lower
    #     maximum = c.prior[variable].domain.events[0][variable].upper
    #     e_in[cb.get("index")] = c.create_range_slider(minimum, maximum,
    #                                                   id={'type': 'i_e_que', 'index': cb.get("index")},
    #                                                   tooltip={"placement": "bottom", "always_visible": False})
    # elif isinstance(variable, random_events.variable.Symbolic):
    #     e_in[cb.get("index")] = dcc.Dropdown(id={"type": "i_e_que", "index": cb.get("index")},
    #                                          options={k: v for k, v in zip(variable.domain[variable],
    #                                                                        variable.domain[variable])},
    #                                          value=list(variable.domain[variable]), multi=True, )
    # elif isinstance(variable, random_events.variable.Integer):
    #     lab = list(variable.domain[variable])
    #     mini = min(lab)
    #     maxi = max(lab)
    #     markings = dict(zip(lab, map(str, lab)))
    #     e_in[cb.get("index")] = c.create_range_slider(minimum=mini - 1, maximum=maxi + 1, value=[mini, maxi],
    #                                                   id={'type': 'i_e_que', 'index': cb.get("index")}, dots=False,
    #                                                   marks=markings,
    #                                                   tooltip={"placement": "bottom", "always_visible": False})
    if len(e_var) - 1 == cb.get("index"):
        return c.add_selector_to_div_button(c.in_use_model, e_var, e_in, e_op, "e_que", cb.get("index") + 1)
    return c.update_free_vars_in_div(c.in_use_model, e_var), e_in, e_op


@callback(
    Output('q_variable_que', 'children'),
    Output('q_input_que', 'children'),
    Output('q_option_que', 'children'),
    Output('e_variable_que', 'children'),
    Output('e_input_que', 'children'),
    Output('e_option_que', 'children'),
    Output('text_l_que', 'children'),
    Output('modal_option_que', 'children'),
    Output('modal_option_que', 'is_open'),
    Output("erg_text_que", "children"),
    Input("erg_b_que", "n_clicks"),
    Input({'type': 'dd_q_que', 'index': ALL}, 'value'),
    Input({'type': 'dd_e_que', 'index': ALL}, 'value'),
    Input({'type': 'b_q_que', 'index': ALL}, 'n_clicks'),
    Input({'type': 'b_e_que', 'index': ALL}, 'n_clicks'),
    Input({'type': 'option_save_que', 'index': ALL}, 'n_clicks'),
    State('q_variable_que', 'children'),
    State('q_input_que', 'children'),
    State('e_variable_que', 'children'),
    State('e_input_que', 'children'),
    State('q_option_que', 'children'),
    State('e_option_que', 'children'),
    State({'type': 'op_i_que', 'index': ALL}, 'value'),
    State("erg_text_que", 'children')
)
def query_router(b_erg, q_dd, e_dd, b_q, b_e, op_s, q_var, q_in, e_var, e_in, q_op, e_op, op_i, erg_out):
    """
        Receives app callback events and manages/redirects these to the correct functions.
    :param q_dd: Query Varibels Names
    :param e_dd: Evidence Variable Names
    :param b_q: Trigger if the Zoom Button in the Query is Pressed
    :param b_e: Trigger if the Zoom Button in the Evidence is Pressed
    :param op_s: Trigger if the Modal parameter from a Zoom should be saved
    :param q_var: Div of the Query Variable
    :param q_in: Div or the Input of Query
    :param e_var: Div of the Evidence Variable
    :param e_in: Div or the Input of Evidence
    :param q_op: Information of whiche Zoom Button was pressed in the Query section
    :param e_op: Information of whiche Zoom Button was pressed in the Evidence section
    :param op_i: The Values choosen in the Zoom Modal
    :return: Query Varibels, Query Input, Evidence Variable, Evidence Input, Text Prefix.
    """
    global modal_var_index
    global modal_type  # 0 = q and 1 = e

    cb = ctx.triggered_id if not None else None
    if cb is None:
        return q_var, q_in, q_op, e_var, e_in, e_op, \
            c.create_prefix_text_query(len_fac_q=len(q_var), len_fac_e=len(e_var)), modal_basic_que, False, erg_out
    # Cateches the cb String it is otherwise a Dic but b are Strings!
    elif cb == "erg_b_que":
        return q_var, q_in, q_op, e_var, e_in, e_op, \
            c.create_prefix_text_query(len_fac_q=len(q_var), len_fac_e=len(e_var)), modal_basic_que, False, \
            infer(c.value_getter_from_children(q_var), c.value_getter_from_children(q_in),
                  c.value_getter_from_children(e_var), c.value_getter_from_children(e_in))

    elif cb.get("type") == "dd_q_que":
        return *query_gen(q_dd, q_var, q_in, q_op), e_var, e_in, e_op, \
            c.create_prefix_text_query(len_fac_q=len(q_var), len_fac_e=len(e_var)), modal_basic_que, False, erg_out
    elif cb.get("type") == "dd_e_que":

        return q_var, q_in, q_op, *evid_gen(e_dd, e_var, e_in, e_op), \
            c.create_prefix_text_query(len_fac_q=len(q_var), len_fac_e=len(e_var)), modal_basic_que, False, erg_out
    elif cb.get("type") == "b_e_que" and e_dd[cb.get("index")] != []:

        modal_var_index = cb.get("index")
        modal_type = 1
        variable = c.vardict[e_dd[cb.get("index")]]
        modal_body = List
        if isinstance(variable, random_events.variable.Continuous) or isinstance(variable, random_events.variable.Integer):
            modal_body = c.generate_modal_option(model=c.in_use_model, var=e_var[cb.get("index")]['props']['value'],
                                                 value=[e_in[cb.get("index")]['props']['min'],
                                                        e_in[cb.get("index")]['props']['max']],
                                                 priors=c.prior, id="_que")

        elif isinstance(variable, random_events.variable.Symbolic):
            modal_body = c.generate_modal_option(model=c.in_use_model, var=e_var[cb.get("index")]['props']['value'],
                                                 value=e_in[cb.get("index")]['props'].get('value'), priors=c.prior,
                                                 id="_que")
        return q_var, q_in, q_op, e_var, e_in, e_op, \
            c.create_prefix_text_query(len_fac_q=len(q_var), len_fac_e=len(e_var)), modal_body, True, erg_out
    elif cb.get("type") == "b_q_que" and q_dd[cb.get("index")] != []:

        modal_var_index = cb.get("index")
        modal_type = 0
        variable = c.vardict[q_dd[cb.get("index")]]
        if isinstance(variable, random_events.variable.Continuous):
            modal_body = c.generate_modal_option(model=c.in_use_model, var=q_var[cb.get("index")]['props']['value'],
                                                 value=[q_in[cb.get("index")]['props']['min'],
                                                        q_in[cb.get("index")]['props']['max']],
                                                 priors=c.prior, id="_que")
        elif isinstance(variable, random_events.variable.Symbolic):
            modal_body = c.generate_modal_option(model=c.in_use_model, var=q_var[cb.get("index")]['props']['value'],
                                                 value=q_in[cb.get("index")]['props'].get('value'), priors=c.prior,
                                                 id="_que")
        elif isinstance(variable, random_events.variable.Integer):
            lab = []
            for seti in c.prior[variable].support.simple_sets[0][variable]:
                lab.extend(list(range(int(seti.lower), int(seti.upper + 1))))
            mini = min(lab)
            maxi = max(lab)
            markings = dict(zip(lab, map(str, lab)))
            modal_body = c.generate_modal_option(model=c.in_use_model, var=q_var[cb.get("index")]['props']['value'],
                                                 value=[mini, maxi],
                                                 priors=c.prior, id="_que")

        return q_var, q_in, q_op, e_var, e_in, e_op, \
            c.create_prefix_text_query(len_fac_q=len(q_var), len_fac_e=len(e_var)), modal_body, True, erg_out
    elif cb.get("type") == "option_save_que":
        new_vals = List
        variable = c.vardict[q_dd[cb.get("index")]] if modal_type == 0 else c.vardict[
            e_dd[cb.get("index")]]
        if isinstance(variable, random_events.variable.Continuous) or isinstance(variable, random_events.variable.Integer):
            new_vals = c.fuse_overlapping_range(op_i)
        else:
            new_vals = op_i[0]  # is List of a List
        if modal_type == 1:
            # print("-"*40)
            # print(e_in)
            e_in[modal_var_index]['props']['value'] = new_vals
            e_in[modal_var_index]['props']['drag_value'] = new_vals
            return q_var, q_in, q_op, e_var, e_in, e_op, \
                c.create_prefix_text_query(len_fac_q=len(q_var), len_fac_e=len(e_var)), modal_basic_que, False, erg_out
        else:
            q_in[modal_var_index]['props']['value'] = new_vals
            q_in[modal_var_index]['props']['drag_value'] = new_vals
            return q_var, q_in, q_op, e_var, e_in, e_op, \
                c.create_prefix_text_query(len_fac_q=len(q_var), len_fac_e=len(e_var)), modal_basic_que, False, erg_out
    else:
        return q_var, q_in, q_op, e_var, e_in, e_op, \
            c.create_prefix_text_query(len_fac_q=len(q_var), len_fac_e=len(e_var)), modal_basic_que, False, erg_out


@callback(
    Output("modal_input_que", "children"),
    Input("op_add_que", "n_clicks"),
    Input({'type': 'op_i_que', 'index': ALL}, 'value'),
    State("modal_input_que", "children"),
    State({'type': 'dd_e_que', 'index': ALL}, 'value'),
    State({'type': 'dd_q_que', 'index': ALL}, 'value'),
)
def modal_router(op, op_i, m_bod, dd_e, dd_q):
    """
        Recessive all App Calls that are change the Modal for the zoom Function
    :param op: Trigger to add More Input Option by Numeric Variabel
    :param op_i: Trigger to update Chance for the Chosen values
    :param m_bod: The State of the Modal
    :param dd_e: div withe the chosen values in the Evidence Section
    :param dd_q: div withe the chosen values in the Query Section
    :return: update Modal Body for the Zoom
    """
    cb = ctx.triggered_id if not None else None
    if cb is None:
        return m_bod
    global modal_var_index
    global modal_type
    dd = dd_e if modal_type == 1 else dd_q
    var = dd[modal_var_index]
    variable = c.vardict[var]
    if not isinstance(m_bod, list):
        m_in_new = [m_bod]
    else:
        m_in_new = m_bod
    if cb == "op_add_que":
        index = m_in_new[-2]['props']['children'][0]['props']['children'][1]['props']['id']['index']
        # var_type = m_in_new[1]['props']['children'][0]['props']['children'][1]['type']
        return c.modal_add_input(body=m_in_new, id_type='op_i_que', index=index, var=var)
    else:  # if cb.get("type") == "op_i"
        index = cb.get("index")
        return c.modal_save_input(body=m_in_new, index=index, var=var)


def infer(q_var, q_in, e_var, e_in):
    """
        Calculates with pm the Probilty of query and evidence
    :param q_var: Div of the Query Variable
    :param q_in: Div or the Input of Query
    :param e_var: Div of the Evidence Variable
    :param e_in: Div or the Input of Evidence
    :return: Probability as String
    """
    cb = ctx.triggered_id if not None else None
    if cb is None:
        return ""

    query = c.div_to_event(c.in_use_model, q_var, q_in)
    evidence = c.div_to_event(c.in_use_model, e_var, e_in)
    print(query)
    print(evidence, "-"*20)

    try:
        c.in_use_model: ProbabilisticCircuit
        conditional_model, p_e = c.in_use_model.truncated(evidence.as_composite_set())
        p_q = conditional_model.probability(query.as_composite_set())
    except Exception as e:
        print(e)
        return "Unsatasfiable"
    return f"P(Q|E) = {round(p_e, 2) * round(p_q, 2)*100}% / {round(p_e * 100, 2)}% = {round(p_q * 100, 2)}%"
r"""
    P(Q|E)
     = 
     \frac{P(Q, E)}{P(E)} = \frac{conditional_probability * evidence_probability}{evidence_probability} = conditional_probability
    """

# def update_time_slot(time, q_v, q_i, q_o, e_v, e_i, e_o, erg):
#     global time_list
#     global old_time
#     # Save the Now Value in Postion Old_time
#     now_value = {'q_v': q_v, "q_i": q_i, "q_o": q_o, 'e_v': e_v, "e_i": e_i, "e_o": e_o, "erg": erg}
#     # Update old_time
#     time_list[old_time] = now_value
#     old_time = time


# @callback(
#     Output("que_time", "max"),
#     Input("que_timer_plus", "n_clicks"),
#     Input("que_timer_minus", "n_clicks"),
#     State("que_time", "value"),
#     State("que_time", "max")
# )
# def button_time(p_b, m_b, q_value, q_max):
#     global time_list
#     global old_time
#     cb = ctx.triggered_id if not None else None
#     if cb is not None and cb == "que_timer_plus":
#         new_dic = dict()
#         new_dic.update({"q_v": [
#             dcc.Dropdown(id={'type': 'dd_q_que', 'index': 0}, options=sorted(c.vardict.keys()), className="")]})
#         new_dic.update({"q_i": [dcc.Dropdown(id={'type': 'i_q_que', 'index': 0}, disabled=True)]})
#         new_dic.update({"q_o": [
#             dbc.Button("👁️", id=dict(type='b_q_que', index=0), disabled=True, n_clicks=0, className="", size="sm")]})
#         new_dic.update({"e_v": [dcc.Dropdown(id={'type': 'dd_e_que', 'index': 0}, options=sorted(c.vardict.keys()))]})
#         new_dic.update({"e_i": [dcc.Dropdown(id={'type': 'i_e_que', 'index': 0}, disabled=True)]})
#         new_dic.update({"e_o": [
#             dbc.Button("👁️", id=dict(type='b_e_que', index=0), disabled=True, n_clicks=0, className="", size="sm")]})
#         new_dic.update({"erg": []})
#
#         if q_max < len(time_list):
#             time_list[q_max] = new_dic
#         else:
#             time_list.append(new_dic)
#
#         return q_max + 1
#     else:
#
#         if q_value[0] == q_max:
#             return q_max
#         else:
#             return q_max - 1 if q_max > 1 else 1
