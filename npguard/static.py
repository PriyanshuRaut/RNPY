import ast

class RiskVisitor(ast.NodeVisitor):
    def __init__(self):
        self.risks = []

    def visit_BinOp(self, node):
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            self.risks.append({
                "line": node.lineno,
                "risk": "chained_binary_op",
                "reason": "Binary operation may allocate temporary arrays"
            })
        self.generic_visit(node)

def analyze_source(source: str):
    tree = ast.parse(source)
    v = RiskVisitor()
    v.visit(tree)
    return v.risks
