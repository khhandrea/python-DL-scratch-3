import heapq
import os
import subprocess

def _dot_var(v, verbose=False):
    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)

    return f'{id(v)} [label="{name}", color=orange, style=filled]\n'

def _dot_func(f):
    ret = f'{id(f)} [label="{f.__class__.__name__}", color=lightblue, style=filled, shape=box]\n'

    # for edge
    for x in f.inputs:
        ret += f'{id(x)} -> {id(f)}\n'
    for y in f.outputs:  # y is weakref
        ret += f'{id(x)} -> {id(y())}\n'
    return ret

def get_dot_graph(output, verbose=True):
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            heapq.heappush(funcs, (-f.generation, f))
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()[1]
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)

    return f'digraph g {{\n{txt}}}'

def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose)

    tmp_dir = os.path.join(os.path.expanduser('~'), '.dezero')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    extension = os.path.splitext(to_file)[1][1:]
    cmd = f'dot {graph_path} -T {extension} -o {to_file}'
    subprocess.run(cmd, shell=True)