from .patchable_graph_interpreter import PatchableGraphInterpreter
from torch.fx import GraphModule


class CompiledGraphModule(GraphModule):

    def _interpret(self, *args, **kwargs):
        interpreter = PatchableGraphInterpreter(self)

        # Interpreter can't handle kwargs, so we need to put them in canonical order.
        # Placeholder nodes have the argument name and default value in the "target" and "args"
        # attributes, respectively.
        graph_args = [n for n in self.graph.nodes if n.op == "placeholder"]
        canonical_args = []
        # Since Python allows passing positional args as keywords, we need to start with the
        # positional args that were actually passed.
        for arg in args:
            canonical_args.append(arg)
            graph_args = graph_args[1:]
        for graph_arg in graph_args:
            if graph_arg.target in kwargs:
                canonical_args.append(kwargs[graph_arg.target])
            else:
                canonical_args.append(graph_arg.args[0])

        return interpreter.run(*canonical_args)

    def recompile(self):
        python_code = super().recompile()

        # Each instance of GraphModule is actually an instance of a unique dynamically generated
        # class, which gets its forward() method overwritten during compile. Since we need to
        # inject our custom interpreter, we need to override that.
        cls = type(self)
        cls.forward = CompiledGraphModule._interpret

        return python_code
