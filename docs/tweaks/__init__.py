import furo
from bs4 import BeautifulSoup


# h/t workaround in https://github.com/sphinx-doc/sphinx/issues/10785
def resolve_type_aliases(app, env, node, contnode):
    aliases = app.config.autodoc_type_aliases
    if node["refdomain"] == "py" and node["reftype"] == "class" and node["reftarget"] in aliases:
        return app.env.get_domain("py").resolve_xref(
            env, node["refdoc"], app.builder, "data", node["reftarget"], node, contnode
        )


PREFIXES_TO_STRIP = ("torch.nn", "torch.fx", "bitsandbytes.nn")


def dequalify_intersphinx(app, doctree, docname):
    from docutils.nodes import NodeVisitor, Text

    class Visitor(NodeVisitor):
        def dispatch_visit(self, node):
            # Bit of a hammer, but this seems like the simplest way to get intersphinx to not
            # qualify cross-references on a case-by-case basis. We have no easy hook to modify
            # the text it generates because resolve_type_aliases is all-or-nothing--our own hook
            # would always be either too late or too early.
            if any(str(node).startswith(p) for p in PREFIXES_TO_STRIP):
                node.parent.children = [Text(node.split(".")[-1])]

    doctree.walk(Visitor(doctree.document))
    pass


def setup(app):
    orig_get_navigation_tree = furo.get_navigation_tree

    def get_navigation_tree(toctree_html):
        furo_html = orig_get_navigation_tree(toctree_html)
        soup = BeautifulSoup(furo_html, "html.parser")
        # Expand all TOC sections by default
        for checkbox in soup.find_all("input", class_="toctree-checkbox", recursive=True):
            checkbox.attrs["checked"] = ""
        # Don't show section collapse button
        for label in soup.find_all("label"):
            label.decompose()

        return str(soup)

    furo.get_navigation_tree = get_navigation_tree
    app.connect("missing-reference", resolve_type_aliases)
    app.connect("doctree-resolved", dequalify_intersphinx)
