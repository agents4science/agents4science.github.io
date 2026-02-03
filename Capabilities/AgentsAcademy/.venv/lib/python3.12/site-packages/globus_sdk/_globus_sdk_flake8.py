from __future__ import annotations

import ast
import typing as t

CODEMAP: dict[str, str] = {
    # SDK001 is necessary for SDK002 enforcement to be easy
    # otherwise, we would have to have a more sophisticated linter which knows about
    # lexical scopes!
    "SDK001": "SDK001 loggers should be named 'log'",
    "SDK002": "SDK002 never use 'log.info'",
}


class Plugin:
    name = "globus-sdk-flake8"
    version = "1.0.0"

    # args to init determine plugin behavior. see:
    # https://flake8.pycqa.org/en/latest/plugin-development/plugin-parameters.html#indicating-desired-data
    #
    # by having "tree" as an init arg, we tell flake8 that we are an AST-handling
    # plugin, run once per file
    def __init__(self, tree: ast.AST) -> None:
        self.tree = tree

    # Plugin.run() is how checks will run. For detail, see implementation of:
    # https://flake8.pycqa.org/en/latest/internal/checker.html#flake8.checker.FileChecker.run_ast_checks
    def run(self) -> t.Iterator[tuple[int, int, str, type]]:
        visitor = SDKVisitor()
        visitor.visit(self.tree)
        for lineno, col, code in visitor.collect:
            yield lineno, col, CODEMAP[code], type(self)


class SDKVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.collect: list[tuple[int, int, str]] = []

    def _record(self, node: ast.expr | ast.stmt, code: str) -> None:
        self.collect.append((node.lineno, node.col_offset, code))

    # see the structure of an assignment:
    #
    # >>> print(ast.dump(ast.parse("""\
    # ... log = logging.getLogger(__name__)
    # ... """), indent=4))
    #
    # Module(
    #     body=[
    #         Assign(
    #             targets=[
    #                 Name(id='log', ctx=Store())],
    #             value=Call(
    #                 func=Attribute(
    #                     value=Name(id='logging', ctx=Load()),
    #                     attr='getLogger',
    #                     ctx=Load()),
    #                 args=[
    #                     Name(id='__name__', ctx=Load())],
    #                 keywords=[]))],
    #     type_ignores=[])
    #
    # we will try to find an optimal path to bail out quickly on most assignments
    def visit_Assign(self, node: ast.Assign) -> None:
        # the value must be a call and must be using attr access in the function call
        # this eliminates bare funcs like `x = foo()` but allows `x = foo.bar()`
        if not (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
        ):
            self.generic_visit(node)
            return
        call_node: ast.Call = node.value
        func_node: ast.Attribute = node.value.func

        # not getLogger? irrelevant!
        # not a name access (e.g., `foo().bar`)? irrelevant!
        # not `getLogger` of `logging` (e.g., `x.getLogger`)? irrelevant! (and weird!)
        if not (
            func_node.attr == "getLogger"
            and isinstance(func_node.value, ast.Name)
            and func_node.value.id == "logging"
        ):
            self.generic_visit(node)
            return

        # the assignee must be a single variable and it must be a name node
        if not (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            self.generic_visit(node)
            return
        name_node: ast.Name = node.targets[0]

        # confirm that the `logging.getLogger` args look right, if not... ignore
        if len(call_node.args) != 1 or not isinstance(call_node.args[0], ast.Name):
            self.generic_visit(node)
            return
        logger_arg: ast.Name = call_node.args[0]

        # now, all data prepared, do the check:
        # - if the argument to `getLogger` is `"__name__"`
        # - and the assignee is not "log"
        # that fails
        #
        # other usages are allowed, e.g. `liblog = logging.getLogger(otherlib_name)
        if name_node.id != "log" and logger_arg.id == "__name__":
            self._record(node, "SDK001")

    # see the structure of a call:
    #
    # >>> print(ast.dump(ast.parse("log.info('foo')"), indent=4))
    # Module(
    #     body=[
    #         Expr(
    #             value=Call(
    #                 func=Attribute(
    #                     value=Name(id='log', ctx=Load()),
    #                     attr='info',
    #                     ctx=Load()),
    #                 args=[
    #                     Constant(value='foo')],
    #                 keywords=[]))],
    #     type_ignores=[])
    #
    def visit_Call(self, node: ast.Call) -> None:
        # if it's not a call to 'x.info(...)', ignore
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "info"):
            self.generic_visit(node)
            return
        func_node: ast.Attribute = node.func

        # if the function was not a method of something named "log", ignore
        if not (isinstance(func_node.value, ast.Name) and func_node.value.id == "log"):
            self.generic_visit(node)
            return

        # nothing left, it failed SDK002!
        self._record(node, "SDK002")
