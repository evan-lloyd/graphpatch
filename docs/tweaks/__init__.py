import furo
from bs4 import BeautifulSoup


# h/t workaround in https://github.com/sphinx-doc/sphinx/issues/10785
def resolve_type_aliases(app, env, node, contnode):
    aliases = app.config.autodoc_type_aliases
    if node["refdomain"] == "py" and node["reftype"] == "class" and node["reftarget"] in aliases:
        return app.env.get_domain("py").resolve_xref(
            env, node["refdoc"], app.builder, "data", node["reftarget"], node, contnode
        )
    elif "NodePath" in node["reftarget"]:
        node["reftype"] = "ref"
        node["refdomain"] = "std"
        node["refexplicit"] = True
        return app.env.get_domain("std").resolve_xref(
            env, node["refdoc"], app.builder, "ref", "node_path", node, contnode
        )


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
