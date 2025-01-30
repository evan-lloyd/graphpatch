import pickle


def find_pickle_error(module):
    """When we get pickling errors, it's almost always because something got left in the FX graph
    meta somewhere. Try to find where it's actually happening by attempting to dump submodules
    bottom-up, until we crash."""

    for meta in module._original_graph.reverse_topological_order():
        print(meta.name)
        if meta.node:
            pickle.dumps(meta.node.meta)
