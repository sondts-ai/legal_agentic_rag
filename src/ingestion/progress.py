from dataclasses import dataclass

STEPS=["load", "clean", "chunk"]

@dataclass
class StepState:
    status:str="pending",
    count:int=0,
    total:int=0

state = {
    step: StepState()
    for step in STEPS
}

def step_start(name,total=0):
    state[name].status = "running"
    state[name].total = total
    state[name].count = 0

def step_update(name,count):
    state[name].count=count

def step_done(name, count=0):
    state[name].status = "done"
    state[name].count = count
    state[name].total = count


def get_state():
    return state

step_start("load", 100)

step_update("load", 50)

print(get_state())

step_done("load", 100)

print(get_state())