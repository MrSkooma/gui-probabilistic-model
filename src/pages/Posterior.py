import dash_bootstrap_components as dbc
import dash
import jpt
from dash import dcc, html, Input, Output, State, ctx, ALL, callback
import components as c
from typing import List
import jpt.distributions.univariate.numeric

global modal_var_index
modal_var_index = -1

global modal_basic_pos
modal_basic_pos = c.gen_modal_basic_id("_pos")

modal_option_pos = c.gen_modal_option_id("_pos")


dash.register_page(__name__)


def layout_pos():
    return dbc.Container(
        [
            dcc.Store(id='memory_pos', data={'time_list': [c.get_default_dic_pos()], 'time': 0}),
            dbc.Row(
                [
                    dbc.Col(html.H1("Posterior", className='text-center mb-4'), width=12),
                ]
            ),

            dbc.Row([
                dbc.Col([dbc.Button("-", n_clicks=0, className="align-self-start mt-0 flex-grow-0", size="sm",
                                    id="pos_timer_minus", style={'verticalAlign': 'top', 'width': '40px'})], width=1,
                        className="ps-0 pe-0 mb-2 mt-0 row row-cols-1 g-1 gy-2 justify-content-end"),
                dbc.Col([dcc.RangeSlider(min=1, max=1, value=[1], step=1, dots=False,
                                         tooltip={"placement": "bottom", "always_visible": False}, id="pos_time")],
                        width=10, className="pe-0 pt-3"),
                dbc.Col(children=[
                    dbc.Button("+", n_clicks=0, className="mt-0 flex-grow-0 align-self-start", size="sm",
                               id="pos_timer_plus", style={'verticalAlign': 'top', 'width': '40px'})],
                    width=1, className="ps-0 pe-0 mb-2 mt-0 row row-cols-1 g-1 gy-2 justify-content-start"),
            ], className="mt-2 d-flex justify-content-center align-items-center"
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
                                dcc.Dropdown(id="text_var_pos", options=sorted(c.in_use_tree.varnames),
                                             value=sorted(c.in_use_tree.varnames),
                                             multi=True, disabled=False)],
                            width=4,
                            className="row row-cols-1 g-1 gy-2 align-items-center border-start border-3 rounded-4 border-secondary"),
                    dbc.Col(id="e_variable_pos",
                            children=[dcc.Dropdown(id={'type': 'dd_e_pos', 'index': 0},
                                                   options=sorted(c.in_use_tree.varnames))],
                            width=2,
                            className="row row-cols-1 g-1 gy-2 align-items-center border-start border-3 border-secondary"),
                    dbc.Col(id="e_input_pos",
                            children=[dcc.Dropdown(id={'type': 'i_e_pos', 'index': 0}, disabled=True)], width=3,
                            className="row row-cols-1 g-1 gy-2 align-items-center"),
                    dbc.Col(id="e_option_pos", children=[
                        dbc.Button("👁️", id=dict(type='b_e_pos', index=0), disabled=True, n_clicks=0,
                                   className="",
                                   size="sm")],
                            className=" row row-cols-1 g-1 gy-2 align-items-center pe-3 ps-1 border-end border-secondary border-3 rounded-4",
                            style={'width': '40px'})
                ], className="justify-content-md-center"
            ),
            dbc.Row(dbc.Button("=", id="erg_b_pos", className="d-grid gap-2 col-3 mt-3 mx-auto", n_clicks=0)),
            dbc.Row(dbc.Col(html.H2("", className='text-center mb-4', id="head_erg_pos"), className="pt-3", width=12)),
            dbc.Row(
                [
                    dbc.Col(dbc.Button("<", id="b_erg_pre_pos", n_clicks=0, disabled=True),
                            className="d-flex justify-content-end align-self-stretch"),
                    dbc.Col(children=[], id="erg_pos", className="", width=8),
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
    Output('head_erg_pos', 'children'),
    Output('erg_pos', 'children'),
    Output('b_erg_pre_pos', 'disabled'),
    Output('b_erg_next_pos', 'disabled'),
    Output('memory_pos', 'data'),
    Output("pos_time", "max"),
    Input('pos_time', 'value'),
    Input({'type': 'dd_e_pos', 'index': ALL}, 'value'),
    Input({'type': 'b_e_pos', 'index': ALL}, 'n_clicks'),
    Input({'type': 'option_save_pos', 'index': ALL}, 'n_clicks'),
    Input('erg_b_pos', 'n_clicks'),
    Input('b_erg_pre_pos', 'n_clicks'),
    Input('b_erg_next_pos', 'n_clicks'),
    Input("pos_timer_plus", "n_clicks"),
    Input("pos_timer_minus", "n_clicks"),
    State('e_variable_pos', 'children'),
    State('e_input_pos', 'children'),
    State('q_variable_pos', 'children'),
    State('e_option_pos', 'children'),
    State({'type': 'op_i_pos', 'index': ALL}, 'value'),
    State('head_erg_pos', 'children'),
    State('erg_pos', 'children'),
    State('b_erg_pre_pos', 'disabled'),
    State('b_erg_next_pos', 'disabled'),
    State("memory_pos", 'data'),
    State("pos_time", "max")
)
def post_router(time, dd_vals, b_e, op_s, n1, n2, n3, n4, n5, e_var, e_in, q_var, e_op, op_i, h_erg, erg, b_erg_pre,
                b_erg_next, mem, t_max):
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

    now_men = (mem.get('time_list'))[mem.get('time')]

    cb = ctx.triggered_id if not None else None
    print("cb ", cb)
    if cb is None:
        return (e_var, e_in, e_op, c.create_prefix_text_query(len(e_var), len(e_var)), q_var, modal_basic_pos, False,
                h_erg, erg, b_erg_pre, b_erg_next, mem, t_max)
    elif cb == "pos_time":
        return *update_time_slot((time[0] - 1), e_var, e_in, e_op, q_var, mem), t_max
    elif cb in ['pos_timer_plus', 'pos_timer_minus']:
        new_t_max, new_mem = button_time(cb, time, t_max, mem)
        return (e_var, e_in, e_op, c.create_prefix_text_query(len(e_var), len(e_var)), q_var, modal_basic_pos, False,
                h_erg, erg, b_erg_pre, b_erg_next, new_mem, new_t_max)
    elif cb in ["erg_pos", 'b_erg_pre_pos', 'b_erg_next_pos', 'erg_b_pos']:
        h_erg, erg, b_l_erg, b_r_erg, n_mem = erg_controller((time[0] - 1), cb, e_var, e_in, q_var, now_men)
        mem.get('time_list')[mem.get('time')].update(n_mem)
        return (e_var, e_in, e_op, c.create_prefix_text_query(len(e_var), len(e_var)), q_var, modal_basic_pos, False,
                h_erg, erg, b_l_erg, b_r_erg, mem, t_max)
    elif cb.get("type") == "dd_e_pos":
        if dd_vals[cb.get("index")] is None:
            return (*c.del_selector_from_div_button(c.in_use_tree, e_var, e_in, e_op, cb.get("index")),
                    c.create_prefix_text_query(len(e_var), len(e_var)), q_var, modal_basic_pos, False, h_erg, erg,
                    b_erg_pre,
                    b_erg_next, mem, t_max)

        variable = c.in_use_tree.varnames[dd_vals[cb.get("index")]]
        if variable.numeric:

            minimum = c.priors[variable.name].cdf.intervals[0].upper
            maximum = c.priors[variable].cdf.intervals[-1].lower
            e_in[cb.get("index")] = c.create_range_slider(minimum, maximum,
                                                          id={'type': 'i_e_pos', 'index': cb.get("index")},
                                                          dots=False,
                                                          tooltip={"placement": "bottom", "always_visible": False})

        elif variable.symbolic:
            e_in[cb.get("index")] = dcc.Dropdown(id={"type": "i_e_pos", "index": cb.get("index")},
                                                 options={k: v for k, v in zip(variable.domain.labels.values(),
                                                                               variable.domain.labels.values())},
                                                 value=list(variable.domain.labels.values()),
                                                 multi=True, )
        elif variable.integer:
            lab = list(variable.domain.labels.values())
            mini = min(lab)
            maxi = max(lab)
            markings = dict(zip(lab, map(str, lab)))
            e_in[cb.get("index")] = c.create_range_slider(minimum=mini - 1, maximum=maxi + 1, value=[mini, maxi],
                                                          id={'type': 'i_e_que', 'index': cb.get("index")}, dots=False,
                                                          marks=markings,
                                                          tooltip={"placement": "bottom", "always_visible": False})

        if len(e_var) - 1 == cb.get("index"):
            return (*c.add_selector_to_div_button(c.in_use_tree, e_var, e_in, e_op, "e_pos", cb.get("index") + 1),
                    c.create_prefix_text_query(len(e_var), len(e_var)), q_var, modal_basic_pos, False, h_erg, erg,
                    b_erg_pre,
                    b_erg_next, mem, t_max)
    elif cb.get("type") == "b_e_pos" and dd_vals[cb.get("index")] != []:
        # Dont Like dont know to do it otherwise
        global modal_var_index
        modal_var_index = cb.get("index")
        variable = c.in_use_tree.varnames[dd_vals[cb.get("index")]]
        modal_body = List
        if variable.numeric:
            modal_body = c.generate_modal_option(model=c.in_use_tree, var=e_var[cb.get("index")]['props']['value'],
                                                 value=[e_in[cb.get("index")]['props']['min'],
                                                        e_in[cb.get("index")]['props']['max']],
                                                 priors=c.priors, id="_pos")
        elif variable.symbolic or variable.integer:
            modal_body = c.generate_modal_option(model=c.in_use_tree, var=e_var[cb.get("index")]['props']['value'],
                                                 value=e_in[cb.get("index")]['props'].get('value'), priors=c.priors,
                                                 id="_pos")

        return (e_var, e_in, e_op, c.create_prefix_text_query(len(e_var), len(e_var)), q_var, modal_body, True,
                h_erg, erg, b_erg_pre, b_erg_next, mem, t_max)
    elif cb.get("type") == "option_save_pos":
        variable = c.in_use_tree.varnames[dd_vals[cb.get("index")]]
        new_vals = List
        if variable.numeric or variable.integer:
            new_vals = c.fuse_overlapping_range(op_i)
        else:
            new_vals = op_i[0]  # is List of a List
        e_in[modal_var_index]['props']['value'] = new_vals
        e_in[modal_var_index]['props']['drag_value'] = new_vals
        return (e_var, e_in, e_op, c.create_prefix_text_query(len(e_var), len(e_var)), q_var, modal_basic_pos, False,
                h_erg, erg, b_erg_pre, b_erg_next, mem, t_max)

    return (c.update_free_vars_in_div(c.in_use_tree, e_var), e_in, e_op,
            c.create_prefix_text_query(len(e_var), len(e_var)), q_var, modal_basic_pos, False,
            h_erg, erg, b_erg_pre, b_erg_next, mem, t_max)


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
    variable = c.in_use_tree.varnames[var]
    if not isinstance(m_bod, list):
        m_in_new = [m_bod]
    else:
        m_in_new = m_bod
    if cb == "op_add_pos":
        index = m_in_new[-2]['props']['children'][0]['props']['children'][1]['props']['id']['index']
        type = m_in_new[1]['props']['children'][0]['props']['children'][1]['type']
        if variable.numeric:

            mini = m_in_new[1]['props']['children'][0]['props']['children'][1]['props']['min']
            maxi = m_in_new[1]['props']['children'][0]['props']['children'][1]['props']['max']
            range_string = html.Div(f"Range {index + 2}",
                                    style=dict(color=c.color_list_modal[(index + 1) % (len(c.color_list_modal) - 1)]))
            n_slider = c.create_range_slider(minimum=mini, maximum=maxi, id={'type': 'op_i_pos', 'index': index + 1},
                                             value=[mini, maxi], dots=False,
                                             tooltip={"placement": "bottom", "always_visible": False},
                                             className="flex-fill")
            var_map = c.div_to_variablemap(c.in_use_tree, [var], [[mini, maxi]])
            prob = c.in_use_tree.infer(var_map, {})

            prob_div = html.Div(f"{prob}",
                                style=dict(color=c.color_list_modal[(index + 1) % (len(c.color_list_modal) - 1)]))
            m_in_new.insert(len(m_in_new) - 1, dbc.Row([
                html.Div([range_string, n_slider, prob_div],
                         id=f"modal_color_{(index + 1) % (len(c.color_list_modal) - 1)}",
                         className="d-flex flex-nowrap justify-content-center ps-2")
            ], className="d-flex justify-content-center"))
            return m_in_new
        elif variable.integer:
            lab = list(variable.domain.labels.values())
            mini = min(lab)
            maxi = max(lab)
            markings = dict(zip(lab, map(str, lab)))
            range_string = html.Div(f"Range {index + 2}",
                                    style=dict(color=c.color_list_modal[(index + 1) % (len(c.color_list_modal) - 1)]))
            n_slider = c.create_range_slider(minimum=mini, maximum=maxi, value=[mini, maxi]
                                             , id={'type': 'op_i_pos', 'index': index + 1}, dots=False,
                                             marks=markings,
                                             tooltip={"placement": "bottom", "always_visible": False},
                                             className="flex-fill")
            var_map = c.div_to_variablemap(c.in_use_tree, [var], [[mini, maxi]])
            prob = c.in_use_tree.infer(var_map, {})
            prob_div = html.Div(f"{prob}",
                                style=dict(color=c.color_list_modal[(index + 1) % (len(c.color_list_modal) - 1)]))
            m_in_new.insert(len(m_in_new) - 1, dbc.Row([
                html.Div([range_string, n_slider, prob_div],
                         id=f"modal_color_{(index + 1) % (len(c.color_list_modal) - 1)}",
                         className="d-flex flex-nowrap justify-content-center ps-2")
            ], className="d-flex justify-content-center"))
            return m_in_new
        else:
            # Sollte nicht Triggerbar sein, da bei DDMenu der +Buttone nicht Aktiv ist
            return m_in_new
    else:  # if cb.get("type") == "op_i"
        id = cb.get("index")
        value = m_in_new[id + 1]['props']['children'][0]['props']['children'][1]['props']['value']
        var_map = c.div_to_variablemap(c.in_use_tree, [var], [value])
        prob = c.in_use_tree.infer(var_map, {})
        prob_div = html.Div(f"{prob}", style=dict(color=c.color_list_modal[id % (len(c.color_list_modal) - 1)]))
        m_in_new[id + 1]['props']['children'][0]['props']['children'][2] = prob_div
        return m_in_new


# @callback(
#     Output('head_erg_pos', 'children'),
#     Output('erg_pos', 'children'),
#     Output('b_erg_pre_pos', 'disabled'),
#     Output('b_erg_next_pos', 'disabled'),
#     Input('pos_time', 'value'),
#     Input('erg_b_pos', 'n_clicks'),
#     Input('b_erg_pre_pos', 'n_clicks'),
#     Input('b_erg_next_pos', 'n_clicks'),
#     State({'type': 'dd_e_pos', 'index': ALL}, 'value'),
#     State({'type': 'i_e_pos', 'index': ALL}, 'value'),
#     State('q_variable_pos', 'children'),
# )
def erg_controller(time, cb, e_var, e_in, q_var, now_mem):
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
    page = now_mem.get('page')
    vals = q_var[0]['props']['value']

    '''
            leaf.distributions = VariableMap(
            {
                tree.varnames[v]: tree.varnames[v].domain.from_json(d) for v, d in data['distributions'].items()
            }
    '''
    d_result = {}
    # if now_mem.get('result') != {}:
        # var_dic = {}
        # for v, d in now_mem.get('result').items():
        #      var_dic.update({c.in_use_tree.varnames[v]: jpt.distributions.univariate.numeric.Numeric.from_json(d)})
        # d_result = jpt.variables.VariableMap(variables=iter(c.in_use_tree._variables), data=var_dic)
    #     c.in_use_tree.to_json()
        # evidence_dict = c.div_to_variablemap(c.in_use_tree, c.value_getter_from_children(now_mem.get('e_var')),
        #                                      c.value_getter_from_children(now_mem.get('e_in')))
        # try:
        #     d_result = c.in_use_tree.posterior(evidence=evidence_dict)
        # except:
        #     d_result = {}
        # for v, d in now_mem.get('result').items():
        #     try:
        #         d_result.update({c.in_use_tree.varnames[v], c.in_use_tree.varnames[v].domain.from_json(d)})
        #     except:
        #         print(v, d)
        #         exit(-1)
    d_result = {} if now_mem.get('result') == {} else jpt.variables.VariableMap(
        {
            c.in_use_tree.varnames[v]: c.in_use_tree.varnames[v].domain.from_json(d) for v, d in now_mem.get('result').items()
        }
    )

    #(jpt.variables.VariableMap.from_json(variables=iter(c.in_use_tree._variables), d=now_mem.get('result'), typ=jpt.distributions.univariate.numeric.Numeric))
    if cb is None:
        return [], [], True, True
    if cb == "b_erg_pre_pos":
        page -= 1
        if page == 0:
            now_mem.update({'page': page})
            return vals[page], plot_post(vals, page, d_result), True, False, now_mem
        else:
            now_mem.update({'page': page})
            return vals[page], plot_post(vals, page, d_result), False, False, now_mem
    elif cb == "b_erg_next_pos":
        page += 1
        if len(vals) > page + 1:
            now_mem.update({'page': page})
            return vals[page], plot_post(vals, page, d_result), False, False, now_mem
        else:
            now_mem.update({'page': page})
            return vals[page], plot_post(vals, page, d_result), False, True, now_mem
    elif cb == "pos_time":
        page = now_mem[time].get("page")
        if type(now_mem[time].get('q_var')[0]) is type(dcc.Dropdown()):
            return [], [], True, True, now_mem
        else:
            left_bool = False if page > 0 else True
            n_vars = now_mem[time].get('q_var')[0]['props']['value']
            right_bool = False if len(n_vars) > page + 1 else True
        return n_vars[page], plot_post(n_vars, page, d_result), left_bool, right_bool, now_mem

    elif vals == [] or cb == "b_erg_pos":
        return [], [], True, True, now_mem
    else:
        page = 0
        now_mem.update({'page': page})
        evidence_dict = c.div_to_variablemap(c.in_use_tree, c.value_getter_from_children(e_var),
                                             c.value_getter_from_children(e_in))
        try:
            result = c.in_use_tree.posterior(evidence=evidence_dict)
            result: jpt.variables.VariableMap
            # for v in result.items():
            #     print(v, type(v))

            json_result = result.to_json()
            json_result_var = result.to_json()
            print("gere")
            json_result_map = {v: k.to_json() for v, k in result.map.items()}

            # json_result_map = {var: {'quantile': value.to_json()} for var, value in result.map.items()}
            # json_result.update(json_result_map)
            # # json_result.update({"quantile": result.map().to_json()})
            now_mem.update({'result': json_result})
            #----
            now_mem.update({'e_var': e_var, 'e_in': e_in})
            #----
        except Exception as e:
            print("Error was", type(e), e)
            return "", [html.Div("Unsatisfiable", className="fs-1 text text-center pt-3 ")], True, True, now_mem
        if len(vals) > 1:
            return vals[page], plot_post(vals, page, result), True, False, now_mem
        else:
            return vals[page], plot_post(vals, page,  result), True, True, now_mem


def plot_post(vars: List, n: int, result):
    """
        Generates the Plots for a Varibel in Vars postion n
    :param vars: List of Variabel
    :param n: Postion of the Choosen Variabel
    :return:  Plot
    """
    print("Trigger", type(result))
    if vars is None or len(vars) == 0:
        return []
    var_name = vars[n]
    variable = c.in_use_tree.varnames[var_name]
    if variable.numeric:
        return c.plot_numeric_to_div(var_name, result=result)

    elif variable.symbolic:
        return c.plot_symbolic_to_div(var_name, result=result)

    elif variable.integer:
        return c.plot_symbolic_to_div(var_name, result=result)


def update_time_slot(time, e_var, e_in, e_op, q_var, mem):
    old_mem = (mem.get('time_list'))[mem.get('time')]
    # Save the Now Value in Position Old_time
    now_value = {'e_var': e_var, "e_in": e_in, "e_op": e_op, 'q_var': q_var, 'page': old_mem.get('page'), 'result': old_mem.get('result')}
    # Update old_time
    mem.get('time_list')[mem.get('time')].update(now_value)
    mem.update({'time': time})

    n_value = mem.get('time_list')[time]
    val_n = n_value.get('q_var')[0]["props"]["value"]
    page_now = n_value.get('page')
    print("page", page_now)
    b_erg_pre_n = True if page_now == 0 else False
    b_erg_next_n = True if len(val_n) > page_now + 1 else False
    n_result = {}
    if n_value.get('result') != {}:
        evidence_dict = c.div_to_variablemap(c.in_use_tree, c.value_getter_from_children(n_value.get('e_var')),
                                             c.value_getter_from_children(n_value.get('e_in')))
        try:
            n_result = c.in_use_tree.posterior(evidence=evidence_dict)
        except:
            n_result = {}
    erg = [] if n_result == {} else plot_post(val_n, time, n_result)
    h_erg = "" if erg == [] else val_n[page_now]
    pre_text = c.create_prefix_text_query(len(e_var), len(e_var))

    return (n_value.get('e_var'), n_value.get('e_in'), n_value.get('e_op'), pre_text, n_value.get('q_var'),
            modal_basic_pos, False, h_erg, erg, b_erg_pre_n, b_erg_next_n, mem)

# @callback(
#     Output("pos_time", "max"),
#     Input("pos_timer_plus", "n_clicks"),
#     Input("pos_timer_minus", "n_clicks"),
#     State("pos_time", "value"),
#     State("pos_time", "max")
# )
def button_time(cb, value, max_time, mem):
    time_list = mem.get('time_list')
    if cb is not None and cb == "pos_timer_plus":
        new_dic = c.get_default_dic_pos().copy()


        if max_time < len(time_list):
            time_list[max_time] = new_dic
        else:
            time_list.append(new_dic)
        mem.update({'time_list': time_list})
        return max_time + 1, mem
    else:

        if value[0] == max_time:
            return max_time, mem
        else:
            return max_time - 1 if max_time > 1 else 1 , mem
