from torch.fx import GraphModule
from typing import TYPE_CHECKING

# TODO: resolve circular import more cleanly
if TYPE_CHECKING:
    from ..meta import OutputArgumentIndex


class GraphPatchModule(GraphModule):
    _graphpatch_output_indexes: "OutputArgumentIndex"
