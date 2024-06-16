import math

import networkx as nx
import random_events.variable
from jpt.base.functions import deque
from probabilistic_model.probabilistic_circuit.distributions import SymbolicDistribution, UniformDistribution

from dash import dcc, html
import plotly.graph_objects as go

import dash_bootstrap_components as dbc
from typing import List
import os
import portion
from probabilistic_model.probabilistic_circuit.probabilistic_circuit import ProbabilisticCircuit, DeterministicSumUnit, \
    SmoothSumUnit, DecomposableProductUnit
import random_events.product_algebra as pa
import numpy as np

in_use_model: ProbabilisticCircuit
in_use_model = ProbabilisticCircuit()  # need to be pm model fully
vardict: dict
vardict = dict()
prior: pa.VariableMap
prior = None

color_list_modal = ["#ccff66", "MediumSeaGreen", "Tomato", "SlateBlue", "Violet"]


# default_tree.varnames
# default_tree.features
# default_tree.targets
# FRontpage alle werte
# Nav
# TItle (Name der Datei)
# Number of Paras
# List Varnames in Farben

# LAden BUtten TaskLeiste Breiter machen Button fixen Home func schrieben

# ---MODAL_EYE____
def gen_modal_basic_id(id: str):
    """
        Generates the zoom Modal style Basic Dash Objects withe the Correct ID
    :param id: The Id to Specify the Components for Dash Callbacks
    :return: Zoom Modal Dash Object List
    """
    return [
        dbc.ModalHeader(dbc.ModalTitle('temp')),
        dbc.ModalBody([
            html.Div([dcc.Dropdown(id={'type': f'op_i{id}', 'index': 0}), dbc.Button(id=f"op_add{id}")], id="mod_in")
        ]),
        dbc.ModalFooter(
            [
                dbc.Button("Save", id=dict(type=f"option_save{id}", index=0), className="ms-auto", n_clicks=0)
            ]
        ),
    ]


def gen_modal_option_id(id: str):
    """
        Generates the Zoom Modal Obtions where the Inteative Components will be set
    :param id: The Id to Specify the Components for Dash Callbacks
    :return: Modal Components withe the base for the Inteactive parts
    """
    return dbc.Modal(
        [
            # #Chidlren? alles Generieren
            dbc.ModalHeader(dbc.ModalTitle('temp'), id="mod_header"),
            dbc.ModalBody([

                dbc.Row(id=f"modal_input{id}", children=[
                    dbc.Col([], id={'type': f'op_i{id}', 'index': 0},
                            className="d-flex flex-nowrap justify-content-center ps-2")
                ], className="d-flex justify-content-center"),
                dbc.Row([
                    dbc.Col([
                        dbc.Button("+", id=f"op_add{id}", className="d-grid gap-2 col-3 mt-3 mx-auto", n_clicks=0,
                                   disabled=True)
                    ], width=6, className="d-grid ps2")
                ]),
                dbc.ModalFooter(
                    [
                        dbc.Button("Save", id=dict(type=f"option_save{id}", index=0), className="ms-auto", n_clicks=0)
                    ]
                ),
            ], )
        ],
        id=f"modal_option{id}", is_open=False, size="xl", backdrop="static"
    )


# ---/MODAL-EYE---

def correct_input_div(variable, priors, id, value=None, **kwargs):
    """
        Generate a Dash Componant for the Varibael
    :param variable: The Variabel wich is being displayed
    :param value:  The Value of the Variable chosen from the User
    :param priors: the variabeln dic of the model vars.
    :param id: the id for dash to call it
    :param kwargs: further specifation for the Dash Componant
    :return: a Dash Componant that displays the variable
    """

    prio = priors[variable].support().simple_sets[0][variable]
    if isinstance(variable, random_events.variable.Continuous):
        minimum = prio.simple_sets[0].lower
        maximum = prio.simple_sets[-1].upper
        value = value if value else [minimum, maximum]
        rang = create_range_slider(minimum, maximum, id=id, value=value, dots=False,
                                   tooltip={"placement": "bottom", "always_visible": False}, **kwargs)
        return rang
    elif isinstance(variable, random_events.variable.Symbolic):
        value = value if value else []

        return dcc.Dropdown(id=id,
                            options={k.value: k.name for k in
                                     prio.simple_sets},
                            value=value, multi=True, **kwargs)
    elif isinstance(variable, random_events.variable.Integer):
        lab = []
        for seti in prio.simple_sets:
            lab.extend(list(range(seti.lower, seti.upper + 1)))
        mini = min(lab)
        maxi = max(lab)
        markings = dict(zip(lab, map(str, lab)))
        value = value if value else []
        return create_range_slider(minimum=mini, maximum=maxi, value=[mini, maxi],
                                   id=id, dots=False,
                                   marks=markings,
                                   tooltip={"placement": "bottom", "always_visible": False}, **kwargs)


# --- MODAL-FUNC ---

def generate_plot_for_variable(variable, result):
    traces = result[variable].plot()
    print(traces)
    fig = go.Figure(traces)
    #fig = go.Figure(layout=dict(title=f"Plot of {variable.name}"))
    # for trace in traces:
    #     fig.add_trace(trace)

    return html.Div([dcc.Graph(figure=fig), html.Div(className="pt-2")])


def generate_modal_option(model: ProbabilisticCircuit, var: str, value: List[str or int or float], priors, id):
    """
        Creates a modal for Zoom for a chosen Variabel, the Style is static
    :param model: the model of the Tree
    :param var: the Variabel wiche will be displayed
    :param value: the User chosen Values from the Varibale
    :param priors: the Priors pre calculatet
    :param id: id from Modal will be modal_input_id because the callbacks cant be duplicated
    :return: Zoom Modal for the Variabel in var
    """
    modal_layout = []
    modal_layout.append(dbc.ModalHeader(dbc.ModalTitle(var)))

    variable = vardict[var]
    result = calculate_posterior_distributions(pa.SimpleEvent(), in_use_model)
    map = div_to_event(model, [var], [value])  #
    map: pa.SimpleEvent
    probs = model.probability(map.as_composite_set())
    is_simbolic = False if isinstance(variable, random_events.variable.Continuous) or isinstance(variable,
                                                                                                 random_events.variable.Integer) else True
    body = dbc.ModalBody(id=f"modal_input{id}", children=[
        dbc.Row([  # Grapicen
            dbc.Col([
                generate_plot_for_variable(variable, result)
            ], width=12),
        ]),
        dbc.Row(children=[
            html.Div([  # Inputs
                html.Div("Range 1" if not is_simbolic else "Dropmenu", style=dict(color=color_list_modal[0])),
                correct_input_div(variable=variable, value=value, priors=priors, id={'type': f'op_i{id}', 'index': 0},
                                  className="d-flex flex-fill"),
                html.Div(f"{round(probs, 5)}", style=dict(color=color_list_modal[0])),
            ], id="modal_color_0", className="d-flex justify-content-evenly ps-2")
        ], className="d-flex justify-content-evenly"),
        dbc.Row([
            dbc.Col([
                dbc.Button("+", id=f"op_add{id}", className="d-grid gap-2 col-3 mt-3 mx-auto", n_clicks=0,
                           disabled=is_simbolic)
            ], width=6, className="d-grid ps2")
        ])
    ])

    foot = dbc.ModalFooter(children=[
        dbc.Button("Save", id=dict(type=f"option_save{id}", index=0), className="ms-auto", n_clicks=0)
    ])
    modal_layout.append(body)
    modal_layout.append(foot)
    return modal_layout


def modal_add_input(body, id_type, index, var):
    variable = vardict[var]
    new_body = body
    if not isinstance(variable, random_events.variable.Continuous) \
            and not isinstance(variable, random_events.variable.Integer):
        return new_body
    elif isinstance(variable, random_events.variable.Continuous):

        mini = new_body[1]['props']['children'][0]['props']['children'][1]['props']['min']
        maxi = new_body[1]['props']['children'][0]['props']['children'][1]['props']['max']
        range_string = html.Div(f"Range {index + 2}",
                                style=dict(color=color_list_modal[(index + 1) % (len(color_list_modal) - 1)]))
        n_slider = create_range_slider(minimum=mini, maximum=maxi, id={'type': id_type, 'index': index + 1},
                                       value=[mini, maxi], dots=False,
                                       tooltip={"placement": "bottom", "always_visible": False},
                                       className="flex-fill")

    elif isinstance(variable, random_events.variable.Integer):
        lab = list(variable.support()[variable])
        mini = min(lab)
        maxi = max(lab)
        markings = dict(zip(lab, map(str, lab)))
        range_string = html.Div(f"Range {index + 2}",
                                style=dict(color=color_list_modal[(index + 1) % (len(color_list_modal) - 1)]))
        n_slider = create_range_slider(minimum=mini, maximum=maxi, value=[mini, maxi]
                                       , id={'type': id_type, 'index': index + 1}, dots=False,
                                       marks=markings,
                                       tooltip={"placement": "bottom", "always_visible": False},
                                       className="flex-fill")
    var_event = div_to_event(in_use_model, [variable.name], [[mini, maxi]])
    prob = in_use_model.probability(var_event.as_composite_set())
    prob_div = html.Div(f"{round(prob, 5)}",
                        style=dict(color=color_list_modal[(index + 1) % (len(color_list_modal) - 1)]))
    new_body.insert(len(new_body) - 1, dbc.Row([
        html.Div([range_string, n_slider, prob_div],
                 id=f"modal_color_{(index + 1) % (len(color_list_modal) - 1)}",
                 className="d-flex flex-nowrap justify-content-center ps-2")
    ], className="d-flex justify-content-center"))
    return new_body


def modal_save_input(body, index, var):
    new_body = body
    value = new_body[index + 1]['props']['children'][0]['props']['children'][1]['props']['value']
    var_event = div_to_event(in_use_model, [var], [value])
    prob = in_use_model.probability(var_event.as_composite_set())
    prob_div = html.Div(f"{round(prob, 5)}", style=dict(color=color_list_modal[index % (len(color_list_modal) - 1)]))
    new_body[index + 1]['props']['children'][0]['props']['children'][2] = prob_div
    return new_body

# --- /MODAL_FUNC ---


def create_range_slider(minimum: float, maximum: float, *args, **kwargs) -> \
        dcc.RangeSlider:
    """
        Generate a RangeSlider that resembles a continuous set.
    :param minimum: lowest number possible in the Range of the slider (left-Side)
    :param maximum: the Highest number possible in the Range of the slider (right-Side)
    :param args: Further styling for plotly dash components
    :param kwargs: Further styling for plotly dash components
    :return: The slider as dcc component
    """
    size = maximum - minimum
    # if size > 0:
    #     minimum -= 0.1*size
    #     maximum += 0.1*size
    # else:
    if maximum == minimum:
        minimum -= 1
        maximum += 1
    if "marks" not in kwargs:
        steps = [x for x in np.arange(minimum, maximum, size / 5)] + [maximum]
        kwargs["marks"] = {x: str(round(x, 2)) for x in steps}

    slider = dcc.RangeSlider(**kwargs, min=minimum, max=maximum, allowCross=False)

    return slider


def fuse_overlapping_range(ranges: List) -> List:
    new_vals = []
    new_list = []
    sor_val = sorted(ranges, key=lambda x: x[0])
    while sor_val != []:
        if len(sor_val) > 1 and sor_val[0][1] >= sor_val[1][0]:
            if sor_val[0][1] >= sor_val[1][1]:
                sor_val.pop(1)
            else:
                sor_val[0] = [sor_val[0][0], sor_val[1][1]]
                sor_val.pop(1)
        else:
            new_vals.append(sor_val[0][0])
            new_vals.append(sor_val[0][1])
            new_list.append(sor_val[0])
            sor_val.pop(0)
    return new_vals


# Just Pray that there no other Strucktur for it or im in a pickel
def value_getter_from_children(children: List[dict]):
    value: List = []
    for i in range(0, len(children) - 1):
        value.append(children[i].get('props').get('value'))
    return value


def div_to_variablemap(model: ProbabilisticCircuit, variables: List,
                       constrains: List) -> pa.VariableMap:
    """
        Transforms variable and Constrains List form the GUI to a VariableMap
    :param model: the JPT model of the Prob. Tree
    :param variables: The list of chosen Variables
    :param constrains:  The list of for the Variables on the same Index
    :return: VariableMap of the Variables with its associated Constraints
    """
    var_dict = pa.VariableMap()
    for variable, constrain in zip(variables, constrains):
        # IF Varaibel is Not Noe but Constrain is None Variabel DOmain Basic
        if variable is None or constrain is None:
            continue
        if isinstance(vardict[variable], random_events.variable.Continuous):
            var_dict[vardict[variable]] = pa.SimpleInterval(constrain[0], constrain[1])
        elif isinstance(vardict[variable], random_events.variable.Integer):
            var_dict[vardict[variable]] = pa.Set([pa.SimpleInterval(round(x)) for x in constrain])
        else:
            var_dict[vardict[variable]] = pa.Set(*constrain)

    return var_dict
    # return jpt.variables.VariableMap([(model.varnames[k], v) for k, v in var_dict.items()])


def div_to_event(model: ProbabilisticCircuit, variables: List,
                 constrains: List) -> pa.SimpleEvent:
    """
        Transforms variable and Constrains List form the GUI to a VariableMap
    :param model: the JPT model of the Prob. Tree
    :param variables: The list of chosen Variables
    :param constrains:  The list of for the Variables on the same Index
    :return: VariableMap of the Variables with its associated Constraints
    """
    var_dict = pa.SimpleEvent()
    for variable, constrain in zip(variables, constrains):
        # IF Varaibel is Not Noe but Constrain is None Variabel DOmain Basic
        if variable is None or constrain is None:
            continue
        if isinstance(vardict[variable], random_events.variable.Continuous):
            var_dict[vardict[variable]] = pa.SimpleInterval(constrain[0], constrain[1])
        elif isinstance(vardict[variable], random_events.variable.Integer):
            var_dict[vardict[variable]] = pa.Set(*[pa.SimpleInterval(round(x)) for x in constrain])
        else:
            constrain_enums = symbolic_to_enum(variable, constrain)
            #print(vardict[variable].domain_type()(0), str(vardict[variable].domain_type()(0)), type(vardict[variable].domain_type()(0)))
            var_dict[vardict[variable]] = pa.Set(*constrain_enums)

    return var_dict
    # return jpt.variables.VariableMap([(model.varnames[k], v) for k, v in var_dict.items()])


def symbolic_to_enum(variable: pa.Variable, values: List) -> List:
    var_enum = vardict[variable].domain_type()
    result = []
    for value in values:
        result.append(var_enum(int(value)))
    return result

def enum_to_symbolic(variable: pa.Variable, values: List):

    var_enum = vardict[variable.name].domain_type()
    result = []
    for name, value in var_enum.__members__.items():
        if value in values:
            result.append(name)
    return result


def mpe_result_to_div(model: ProbabilisticCircuit, res: pa.SimpleEvent, likelihood: float) -> List:
    """
        Generate Visuel Dash Representation for result of the mpe jpt func
    :param res: one of the Results from mpe func
    :param likelihood: The likelihood of the maxima
    :return: Children's List from Dash Components to display the Results in res
    """
    return_div = []

    for variable, restriction in res.items():
        prio = prior[variable].support().simple_sets[0][variable]
        if isinstance(variable, random_events.variable.Integer):
            continue
            value = [x for i in range(0, len(restriction)) for x in (i, i)]
            lab = list(variable.support().labels.values())
            mini = min(lab)
            maxi = max(lab)
            markings = dict(zip(lab, map(str, lab)))
            return_div += [html.Div(
                [dcc.Dropdown(options=[variable.name], value=[variable.name], disabled=True, className="margin10"),
                 create_range_slider(minimum=mini - 1, maximum=maxi + 1, value=value, disabled=True, marks=markings,
                                     dots=False,
                                     className="margin10")]
                , style={"display": "grid", "grid-template-columns": "30% 70%"})]

        if isinstance(variable, random_events.variable.Continuous):
            restriction: pa.Interval
            value = restriction.simple_sets[0]#should alway be 1
            # if portion == type(res[variable]):
            #     for interval in res[variable].intervals:
            #         value += [interval.lower, interval.upper]
            # else:
            #     value += [res[variable].lower, res[variable].upper]
            #Assume always range values in a list data type
            minimum = prio.simple_sets[0].lower
            maximum = prio.simple_sets[-1].upper
            value = [value.lower, value.upper]
            return_div += [html.Div(
                [dcc.Dropdown(options=[variable.name], value=variable.name, disabled=True, className="margin10"),
                 create_range_slider(minimum, maximum, value=value, disabled=True, className="margin10",
                                     tooltip={"placement": "bottom", "always_visible": True})]
                , style={"display": "grid", "grid-template-columns": "30% 70%"})]

        elif isinstance(variable, random_events.variable.Symbolic):
            restriction_list = [a for a in restriction.simple_sets]
            res = enum_to_symbolic(variable, restriction_list)
            return_div += [html.Div(
                [dcc.Dropdown(options=[variable.name], value=variable.name, disabled=True),
                 dcc.Dropdown(
                     options=res,
                     value=res, multi=True, disabled=True, className="ps-3")],
                style={"display": "grid", "grid-template-columns": "30% 70%"})]
        return_div += [html.Div(className="pt-1")]

    return_div = [html.Div([dcc.Dropdown(options=["Likelihood"], value="Likelihood", disabled=True,
                                         className="margin10"),
                            dcc.Dropdown(options=[likelihood], value=likelihood, disabled=True, className="ps-3 pb-2")],
                           id="likelihood", style={"display": "grid", "grid-template-columns": "30% 70%"})] + return_div

    return return_div


def create_prefix_text_query(len_fac_q: int, len_fac_e: int) -> List:
    """
        Creates Dash Style Prefix for the query GUI
    :param len_fac_q:  Length of Query input used for Scaling
    :param len_fac_e:  Length of Evidence input used for Scaling
    :return: Children div for the prefix query GUI
    """
    return [
        html.Div("P ", className="pe-3",
                 style={"width": "50%", "height": "100%",
                        'fontSize': (len_fac_q if len_fac_q >= len_fac_e else len_fac_e) * 20,
                        'padding-top': (len_fac_q * 1 if len_fac_q >= len_fac_e else len_fac_e * 1)}),
    ]


def create_prefix_text_mpe(len_fac: int) -> List:
    """
        Creates Dash Style Prefix for the MPE GUI
    :param len_fac: Length of Evidence input used for Scaling
    :return: Children div for the prefix MPE GUI
    """
    return [
        html.Div("argmax ", className="pe-3",
                 style={'padding-top': 0, 'fontSize': len_fac * 10 if len_fac * 10 < 40 else 25}),
        html.Div("P ", className="ps-3",
                 style={'padding-top': 0, "height": "100%", 'fontSize': len_fac * 15 if len_fac * 15 < 75 else 75}),
    ]


def generate_free_variables_from_div(model: ProbabilisticCircuit, variable_div: List) -> List[str]:
    """
        Peels the names out of variable_div elements and uses generate_free_variables_from_list for the Return
    :param model: the JPT model of the Prob. Tree
    :param variable_div: List of all Variabels that are being Used, in Dash Dropdown Class saved
    :return: Returns List of String from the Names of all not used Variabels.
    """
    variable_list = variable_div

    variables = []
    for v in variable_list:
        if len(v['props']) > 2:
            variables += [v['props'].get('value', [])]
    return generate_free_variables_from_list(model, variables)


def generate_free_variables_from_list(model: ProbabilisticCircuit, variable_list: List[str]) -> List[str]:
    """
        Deletes all used Variable Names out of a List of all Variables Names.
    :param model: the JPT model of the Prob. Tree
    :param variable_list: the List of in use Variable Names
    :return: List of Variable Names that are not in use
    """
    #copy?
    vars_free = vardict.copy()

    for v in variable_list:
        if v != []:
            vars_free.pop(v)
    return list(vars_free.keys())


def update_free_vars_in_div(model: ProbabilisticCircuit, variable_div: List) -> List:
    """
        Updates the Variable Options for a Dash Dropdown for choosing Variables, to all not in use Variables.
    :param model: the JPT model of the Prob. Tree
    :param variable_div: the Div to update the Options
    :return: the Div withe updated variable Options
    """
    variable_list = variable_div
    vars_free = generate_free_variables_from_div(model, variable_list)
    d = dict(a="a", b="b")

    for v in variable_list:
        if len(v['props']) > 2:
            if v['props'].get('value', "NULL") == "NULL":
                v['props']['options'] = vars_free
            else:
                v['props']['options'] = [v['props'].get('value')] + vars_free
    return variable_list


def reduce_index(index, number, list) -> List:
    """
        Reduces the index in id from index in the list about the amount number
    :param index: the start index to decrease the index
    :param number: the amount to decrease
    :param list: the List from Dash Components that should be decreased
    :return: list with the decreased index implemented
    """
    for i in range(index, len(list)):
        list[i]['props']['id']['index'] -= number
    return list


def del_selector_from_div(model: ProbabilisticCircuit, variable_div: List, constrains_div: List, del_index: int) \
        -> (List, List):
    """
        Deletes a Row from the Option + Constrains and Rebuilds all Choices for Variables
    :param model: the JPT model of the Prob. Tree
    :param variable_div: list of Components to Chose Variable in the GUI
    :param constrains_div: list of Components that are the Constraints for the Variables on the Same Index
    :param del_index: the Value on what Position the to delete Row is.
    :return: Variable Children and Constrains Children for the GUI withe Update options
    """
    variable_list = variable_div
    constrains_list = constrains_div

    variable_list = reduce_index(del_index, 1, variable_list)
    constrains_list = reduce_index(del_index, 1, constrains_list)

    variable_list.pop(del_index)
    constrains_list.pop(del_index)

    new_var_list = update_free_vars_in_div(model, variable_list)
    return new_var_list, constrains_list


def del_selector_from_div_button(model: ProbabilisticCircuit, variable_div: List, constrains_div: List,
                                 option_div: List, del_index: int) -> (List, List):
    """
        Deletes a Row from the Option + Constrains and Rebuilds all Choices for Variables
    :param option_div:
    :param model: the JPT model of the Prob. Tree
    :param variable_div: list of Components to Chose Variable in the GUI
    :param constrains_div: list of Components that are the Constraints for the Variables on the Same Index
    :param del_index: the Value on what Position the to delete Row is.
    :return: Variable Children and Constrains Children for the GUI withe Update options
    """

    variable_list = variable_div
    constrains_list = constrains_div
    option_list = option_div

    # if len(variable_list) == 1:
    #     variable_list[0]['props']['value'] = ""
    # else:

    variable_list = reduce_index(del_index, 1, variable_list)
    constrains_list = reduce_index(del_index, 1, constrains_list)
    option_list = reduce_index(del_index, 1, option_list)

    variable_list.pop(del_index)
    constrains_list.pop(del_index)
    option_list.pop(del_index)

    new_var_list = update_free_vars_in_div(model, variable_list)
    option_list[-1]['props']['disabled'] = True
    return new_var_list, constrains_list, option_list


def add_selector_to_div(model: ProbabilisticCircuit, variable_div: List, constrains_div: list, type: str,
                        index: int) \
        -> (List[dcc.Dropdown], List):
    """
        Genrats the Correct Selector Components for the div
    :param model: the JPT model of the Prob. Tree
    :param variable_div: list of Components to Chose Variable in the GUI
    :param constrains_div: list of Components that are the Constraints for the Variables on the Same Index
    :param type: the Type of the Component for the ID
    :param index: the index Number of the Component for the ID
    :return: Variable Children and Constrains Children for the GUI withe one more Row
    """
    variable_list = variable_div
    constrains_list = constrains_div

    variable_list = update_free_vars_in_div(model, variable_list)

    variable_list.append(
        dcc.Dropdown(id={'type': f'dd_{type}', 'index': index},
                     options=variable_list[0]['props']['options'][1:]))
    constrains_list.append(dcc.Dropdown(id={'type': f'i_{type}', 'index': index}, disabled=True))
    return variable_list, constrains_list


# --- Button Func ---
def add_selector_to_div_button(model: ProbabilisticCircuit, variable_div, constrains_div, option_div, type: str,
                               index: int) \
        -> (List[dcc.Dropdown], List, List):
    """
        Genrates teh Selector for the div withe a Button
    :param model: the JPT model of the Prob. Tree
    :param variable_div: list of Components to Chose Variable in the GUI
    :param constrains_div: list of Components that are the Constraints for the Variables on the Same Index
    :param type: the Type of the Component for the ID
    :param index: the index Number of the Component for the ID
    :return: Variable Children and Constrains Children for the GUI withe one more Row
    """
    variable_list = variable_div
    constrains_list = constrains_div
    option_list = option_div

    variable_list = update_free_vars_in_div(model, variable_list)
    option_list[-1]['props']['disabled'] = False

    variable_list.append(
        dcc.Dropdown(id={'type': f'dd_{type}', 'index': index},
                     options=variable_list[0]['props']['options'][1:], className=""))
    constrains_list.append(
        dcc.Dropdown(id={'type': f'i_{type}', 'index': index}, disabled=True, className="", style={'padding-top': 0}))
    option_list.append(
        dbc.Button("👁️", id=dict(type=f'b_{type}', index=index), disabled=True, n_clicks=0, className="",
                   size="sm", style={'width': '40px'}))
    return variable_list, constrains_list, option_list


def reset_gui_button(model: ProbabilisticCircuit, type: str):
    """
        Resets the GUI Parts back to Start + Button
    :param model: the JPT Tree
    :param type: What Type of ID it is
    :return: Clean Start Style of Components for the GUI
    """
    var_div = [dcc.Dropdown(id={'type': f'dd_{type}', 'index': 0}, options=sorted(vardict.keys()))]
    in_div = [dcc.Dropdown(id={'type': f'i_{type}', 'index': 0}, disabled=True)]
    op_div = [dbc.Button("👁️", id=dict(type='b_e', index=0), disabled=True, n_clicks=0, className="me-2",
                         size="sm")]
    return var_div, in_div, op_div


# --- Button Func ---


def reset_gui(model: ProbabilisticCircuit, type: str) -> (List, List):
    """
        Resets the GUI Parts back to Start
    :param model: the JPT Tree
    :param type: What Type of ID it is
    :return: Clean Start Style of Components for the GUI
    """
    var_div = [dcc.Dropdown(id={'type': f'dd_{type}', 'index': 0}, options=sorted(vardict.keys()))]
    in_div = [dcc.Dropdown(id={'type': f'i_{type}', 'index': 0}, disabled=True)]
    return var_div, in_div


# Postierior---

# def plot_symbolic_distribution(distribution: jpt.distributions.univariate.Multinomial) -> go.Bar:
#     """
#         generates a Bar graph for symbolic distribution in jpt.
#     :param distribution: the Distribution for the Bar Diagram
#     :return: the trace of a Bar Diagram for the symbolic variable.
#     """
#     trace = go.Bar(x=list(distribution.labels.keys()), y=distribution._params)  # anstatt keys könnte values sein
#     return trace
#
#
# # TODOO
# # X nach Externe Konvertierenn
# def plot_numeric_pdf(distribution: jpt.distributions.univariate.Numeric, padding=0.1) -> go.Scatter:
#     """
#         generates a jpt plot from a numeric variable
#     :param distribution: the Distribution of the variable for the Plot
#     :param padding: for the ends of the Plot, it is for visibility.
#     :return: scatter plot for the numeric variable
#     """
#     x = []
#     y = []
#
#     for interval, function in zip(distribution.pdf.intervals[1:-1], distribution.pdf.functions[1:-1]):
#         x += [interval.lower, interval.upper, interval.upper]
#         y += [function.value, function.value, None]
#
#     x = [distribution.value2label(x_) for x_ in x]
#     range = x[-1] - x[0]
#     x = [x[0] - (range * padding), x[0], x[0]] + x + [x[-1], x[-1], x[-1] + (range * padding)]
#     y = [0, 0, None] + y + [None, 0, 0]
#
#     trace = go.Scatter(x=x, y=y, name="PDF")
#
#     # generate logarithmic scaled trace
#     log_y = [np.log(y_) if y_ is not None and y_ > 0 else None for y_ in y]
#     log_trace = go.Scatter(x=x, y=log_y, name="Logarithmic PDF", visible='legendonly')
#
#     return trace, log_trace
#
#
# def plot_numeric_cdf(distribution: jpt.distributions.univariate.Numeric, padding=0.1) -> go.Scatter:
#     """
#         generates a cdf plot from a numeric variable
#     :param distribution: the Distribution of the variable for the Plot
#     :param padding: for the ends of the Plot, it is for visibility.
#     :return: scatter plot for the numeric variable
#     """
#     x = []
#     y = []
#     for interval, function in zip(distribution.cdf.intervals[1:], distribution.cdf.functions[1:]):
#         x += [interval.lower]
#         y += [function.eval(interval.lower)]
#
#     x = [distribution.value2label(x_) for x_ in x]
#     range = x[-1] - x[0]
#     if range == 0:
#         range = 1
#
#     x = [x[0] - (range * padding), x[0]] + x + [x[-1] + (range * padding)]
#     y = [0, 0] + y + [1]
#     trace = go.Scatter(x=x, y=y, name="CDF")
#     return trace
#
#
# def plot_numeric_to_div(var_name: List, result) -> List:
#     """
#         Generates a Div where both plots are in for a numeric variable
#     :param var_name: the name of variable that will be plotted
#     :param result: the result generate from jpt.
#     :return: one div withe 2 Plots in.
#     """
#     fig = go.Figure(layout=dict(title=f"Cumulative Density Function of {var_name}"))
#     t = plot_numeric_cdf(result[var_name])
#     fig.add_trace(t)
#     is_dirac = result[var_name].is_dirac_impulse()
#     if not is_dirac:
#         fig2 = go.Figure(layout=dict(title=f"Probability Density Function of {var_name}"))
#         t2, t3 = plot_numeric_pdf(result[var_name])
#         fig2.add_trace(t2)
#         fig2.add_trace(t3)
#
#     arg_max, max_ = result[var_name].mpe()
#     arg_max = result[var_name].value2label(arg_max)
#
#     arg_max = arg_max.simplify()
#     if isinstance(arg_max, jpt.base.intervals.ContinuousSet):
#         arg_max = jpt.base.intervals.RealSet([arg_max])
#
#     for interval in arg_max.intervals:
#         if interval.size() <= 1:
#             continue
#
#         fig.add_trace(go.Scatter(x=[interval.lower, interval.upper, interval.upper, interval.lower],
#                                  y=[0, 0, result[var_name].p(list2interval([-float("inf"), interval.upper])),
#                                     result[var_name].p(list2interval([-float("inf"), interval.lower]))],
#                                  fillcolor="LightSalmon",
#                                  opacity=0.5,
#                                  mode="lines",
#                                  fill="toself", line=dict(width=0),
#                                  name="Max"))
#         if not is_dirac:
#             fig2.add_trace(go.Scatter(x=[interval.lower, interval.upper, interval.upper, interval.lower],
#                                       y=[0, 0, max_, max_],
#                                       fillcolor="LightSalmon",
#                                       opacity=0.5,
#                                       mode="lines",
#                                       fill="toself", line=dict(width=0),
#                                       name="Max"))
#
#     try:
#         expectation = result[var_name].expectation()
#         fig.add_trace(go.Scatter(x=[expectation, expectation], y=[0, 1], name="Exp", mode="lines+markers",
#                                  marker=dict(opacity=[0, 1])))
#     except:
#         pass
#
#     if is_dirac:
#         return html.Div([dcc.Graph(figure=fig), html.Div(className="pt-2")], className="pb-3")
#     else:
#         try:
#             expectation = result[var_name].expectation()
#             fig2.add_trace(go.Scatter(x=[expectation, expectation], y=[0, max_ * 1.1], name="Exp", mode="lines+markers",
#                                       marker=dict(opacity=[0, 1])))
#         except:
#             pass
#         return html.Div([dcc.Graph(figure=fig), html.Div(className="pt-2"), dcc.Graph(figure=fig2)], className="pb-3")
#
#
# def plot_symbolic_to_div(var_name: str, result) -> List:
#     """
#         generates a div where a bar Diagram for a Symbolic Variable.
#     :param var_name: the name of the variable
#     :param result: the result generate from jpt
#     :return: a div withe one bar diagram in it.
#     """
#     arg_max, max_ = result[var_name].mpe()
#     fig = go.Figure(layout=dict(title="Probability Distribution"))
#     lis_x_max = []
#     lis_y_max = []
#     lis_x = []
#     lis_y = []
#     for i in range(0, len(result[var_name].labels.keys())):
#         if result[var_name]._params[i] >= max_:
#             lis_x_max += [list(result[var_name].labels.keys())[i]]
#             lis_y_max += [result[var_name]._params[i]]
#         else:
#             lis_x += [list(result[var_name].labels.keys())[i]]
#             lis_y += [result[var_name]._params[i]]
#
#     lis_x = [result[var_name].value2label(x_) for x_ in lis_x]
#     lis_x_max = [result[var_name].value2label(x_) for x_ in lis_x_max]
#
#     fig.add_trace(go.Bar(y=lis_x_max, x=lis_y_max, name="Max", marker=dict(color="LightSalmon"), orientation="h"))
#     fig.add_trace(go.Bar(y=lis_x, x=lis_y, name="Prob", marker=dict(color="CornflowerBlue"), orientation='h', ))
#     return html.Div([dcc.Graph(figure=fig)], className="pb-3")


def gen_Nav_pages(pages, toIgnoreName):
    """
        Genartes the Navigation Page Links, withe out the toIgnoreNames
    :param pages: All Pages that are in the GUI
    :param toIgnoreName: Names of Pages that shouldnt be displayed (Empty)
    :return: Dash Struct for Navgation of Pages
    """
    nav = [p for p in pages if p['name'].lower() not in [x.lower() for x in toIgnoreName]]
    nav_posi = dict(Home=0, Query=1, Most_Probable_Explanation=3, Posterior=2)
    navs = oder_Nav(nav_posi, nav)
    navItems = []
    for page in navs:
        navItems.append(dbc.NavItem(dbc.NavLink(f"{page['name']}", href=page['relative_path'])))
        # Liste solle Home Query Most probable explanation Posterior Rest sein

    return navItems


def oder_Nav(nav_positions: dict, nav: List):
    # sollte in Kontext gehen ohne wieder holte sortieren
    sor = True
    while sor:
        sor = False
        for index, n in enumerate(nav):
            posi = nav_positions.get(n['name'], -1)
            if posi != index and posi != -1:
                sor = True
                nav[posi], nav[index] = nav[index], nav[posi]

    return nav


def get_default_dic_mpe():
    default_dic = dict()
    default_dic.update(
        {"e_var": [dcc.Dropdown(id={'type': 'dd_e_mpe', 'index': 0}, options=sorted(vardict.keys()))]})
    default_dic.update({"e_in": [dcc.Dropdown(id={'type': 'i_e_mpe', 'index': 0}, disabled=True)]})
    default_dic.update({"e_op": [
        dbc.Button("👁️", id=dict(type='b_e_mpe', index=0), disabled=True, n_clicks=0, className="", size="sm")]})
    default_dic.update({"q_var": [
        dcc.Dropdown(id="text_var_mpe", options=sorted(vardict.keys()), value=sorted(vardict.keys()),
                     multi=True, disabled=True)]})
    default_dic.update({'likelihood': 0.0})
    default_dic.update({'page': 0})
    default_dic.update({'maxima': None})
    default_dic.update({'erg_b': (True, True)})
    return default_dic


def get_default_dic_pos():
    default_dic = dict()
    default_dic.update(
        {"e_var": [dcc.Dropdown(id={'type': 'dd_e_pos', 'index': 0}, options=sorted(vardict.keys()))]})
    default_dic.update({"e_in": [dcc.Dropdown(id={'type': 'i_e_pos', 'index': 0}, disabled=True)]})
    default_dic.update({"e_op": [
        dbc.Button("👁️", id=dict(type='b_e_pos', index=0), disabled=True, n_clicks=0, className="", size="sm")]})
    default_dic.update({"q_var": [dcc.Dropdown(id="text_var_pos", options=sorted(vardict.keys()),
                                               value=sorted(vardict.keys()), multi=True, disabled=False)]})
    default_dic.update({'page': 0})
    default_dic.update({'result': {}})
    return default_dic


# ---- PM NEW STUFF -----

def create_prior_distributions(model: ProbabilisticCircuit):
    prior_distributions = pa.VariableMap()
    for variable in model.variables:
        prior_distributions[variable] = model.marginal([variable])
    return prior_distributions


def calculate_posterior_distributions(evidence: pa.SimpleEvent, model: ProbabilisticCircuit):
    posterior_distributions = pa.VariableMap()

    conditional_model, evidence_probability = model.conditional(evidence.as_composite_set())

    for variable in conditional_model.variables:
        posterior_distributions[variable] = conditional_model.marginal([variable])

    return posterior_distributions


def plot_3d(model: ProbabilisticCircuit):
    graph_to_plot = nx.DiGraph()

    nodes_to_plot_queue = deque([model.root])

    while len(nodes_to_plot_queue) > 0:
        current_node = nodes_to_plot_queue.popleft()
        graph_to_plot.add_node(current_node)
        for edge in model.in_edges(current_node):
            graph_to_plot.add_edge(*edge)

        if len(current_node.variables) == 1:
            continue

        for subcircuit in current_node.subcircuits:
            nodes_to_plot_queue.append(subcircuit)

    # Compute positions using an algorithm (e.g., Kamada-Kawai layout)
    # pos = nx.spring_layout(self, scale=0.5, dim=3, threshold=2.0531, k=2, weight=1, iterations=100)
    pos = nx.spring_layout(graph_to_plot, dim=3, scale=0.5)
    # Extract node positions
    node_positions = {node: pos[node] for node in graph_to_plot.nodes}

    # Create a Plotly figure
    fig = go.Figure(layout=go.Layout(showlegend=False, scene=dict(aspectmode='data')))

    # Add edges
    for edge in graph_to_plot.edges():
        x0, y0, z0 = node_positions[edge[0]]
        x1, y1, z1 = node_positions[edge[1]]
        fig.add_trace(go.Scatter3d(
            x=[x0, x1],
            y=[y0, y1],
            z=[z0, z1],
            mode='lines',
            line=dict(color='black', width=3)
        ))

    for node in graph_to_plot.nodes():
        x, y, z = node_positions[node]
        for mark in get_correct_3d_marker(node):
            fig.add_trace(go.Scatter3d(
                x=[x],
                y=[y],
                z=[z],
                mode='markers',
                marker=mark
            ))

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        )
    )

    fig["layout"].update(margin=dict(l=0, r=0, b=0, t=0))

    return fig


def get_correct_3d_marker(node):
    if isinstance(node, SymbolicDistribution):
        return [dict(size=4, color='blue', opacity=0.5, symbol='square')]
    elif isinstance(node, SmoothSumUnit):
        return [dict(size=6, color='blue', opacity=0.8, symbol='circle'),
                dict(size=4, color='azure', opacity=0.8, symbol='cross')]
    elif isinstance(node, DeterministicSumUnit):
        return [dict(size=6, color='blue', opacity=0.4, symbol='circle'),
                dict(size=4, color='blue', opacity=0.4, symbol='circle'),
                dict(size=4, color='azure', opacity=0.8, symbol='cross')]
    elif isinstance(node, DecomposableProductUnit):
        return [dict(size=6, color='blue', opacity=0.8, symbol='circle'),
                dict(size=4, color='azure', opacity=0.8, symbol='x')]
    elif isinstance(node, UniformDistribution):
        return [dict(size=4, color='blue', opacity=0.5, symbol='square')]
    else:
        raise ValueError("Unknown node type {}".format(type(node)))
