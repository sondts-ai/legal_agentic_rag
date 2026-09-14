from dataclasses import dataclass

STEPS = ["load", "clean", "chunk", "chroma", "bm25", "graph"]

@dataclass
class StepState:
    status: str = "pending"
    count: int = 0
    total: int = 0
    message: str = ""

state = {
    step: StepState()
    for step in STEPS
}

def step_start(name, total=0, message=""):
    state[name].status = "running"
    state[name].total = total
    state[name].count = 0
    state[name].message = message


def step_update(name, count, total=None, message=""):
    state[name].count=count
    if total is not None:
        state[name].total = total
    if message:
        state[name].message = message


def step_done(name, count=0, message=""):
    state[name].status = "done"
    state[name].count = count
    state[name].total = count
    state[name].message = message


def step_skip(name, message=""):
    state[name].status = "skipped"
    state[name].message = message


def pipeline_done():
    return None


def pipeline_error(message):
    return None


def get_state():
    return state