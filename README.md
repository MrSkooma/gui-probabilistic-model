# JPT-GUI

Welcome to the JPT-GUI Repository.

## Prerequisites

This project requires the `cognitive_robot_abstract_machine` workspace to be present as a sibling directory, which provides `pycram-robotics`, `probabilistic_model`, and `random_events`.

## Installation

### Using Poetry (Recommended)

```sh
poetry install
```

### Using pip

```sh
pip install -r requirements.txt
```

Both methods will install the GUI dependencies and the local CRAM packages in editable mode.

## Run

Run the GUI:
```sh
python src/app.py
```
