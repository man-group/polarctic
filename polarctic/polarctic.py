import datetime as dt
import ast
import re
import polars as pl
from arcticdb import Arctic, LibraryOptions, QueryBuilder, LazyDataFrame, OutputFormat
from arcticdb.version_store.library import Library
from arcticdb.version_store.processing import ExpressionNode
from arcticdb_ext.version_store import OperationType
from typing import cast, Iterator, Any, Optional

class PolarsToArcticDBTranslator:
    """
    Translates Polars expressions to ArcticDB QueryBuilder operations.
    
    Usage:
        translator = PolarsToArcticDBTranslator()
        qb = translator.translate(polars_expr, query_builder)
    """

    def translate(self, polars_expr: pl.Expr, query_builder: Any) -> Any:
        """
        Translate a Polars expression string to ArcticDB QueryBuilder operations.
        
        Args:
            polars_expr: String representation of Polars expression
            query_builder: ArcticDB QueryBuilder instance
            
        Returns:
            Modified QueryBuilder instance
        """
        
        # Clean the expression - remove surrounding brackets if present
        expr = str(polars_expr).strip()
        if expr.startswith('[') and expr.endswith(']'):
            expr = expr[1:-1].strip()

        expr = self._replace_square_brackets(expr)
        
        # Preprocess to handle Polars-specific notation like [dyn int: 2]
        expr = self._preprocess_expression(expr)

         # Parse the expression
        try:
            tree = ast.parse(expr, mode='eval')
            expr_node = self._process_node(tree.body)
        except SyntaxError as e:
            raise ValueError(f"Invalid Polars expression: {polars_expr}") from e
        
        return query_builder[expr_node]
        
    def _replace_square_brackets(self, text: str) -> str:
        while True:
            close = text.rfind('])')
            if close == -1:
                break
            open_ = text.rfind('([', 0, close)
            if open_ == -1:
                break
            # Replace the matched ([...]) with (...)
            text = text[:open_] + '(' + text[open_ + 2:close] + ')' + text[close + 2:]
        return text
    
    def _preprocess_expression(self, expr: str) -> str:
        """
        Preprocess Polars expression to handle special notation.
        
        Converts patterns like [dyn int: 2] to just the value (2).
        """
        # Pattern to match [dyn type: value] or [lit type: value]
        pattern = r'[\[\(](dyn|lit)\s+\w+:\s*([^\]\)]+)[\]\)]'
        
        def replace_dynamic(match: re.Match) -> Any:
            return match.group(2).strip()
        
        return re.sub(pattern, replace_dynamic, expr)

    def _process_node(self, node: ast.AST) -> Any:
        """Process an AST node and apply corresponding ArcticDB operation."""

        node_type = type(node)
        match node_type:
            case ast.Call:
                return self._process_call(cast(ast.Call, node))
            case ast.Attribute:
                return self._process_attribute(cast(ast.Attribute, node))
            case ast.Name:
                return cast(ast.Name, node).id
            case ast.Constant:
                return cast(ast.Constant, node).value
            case ast.Compare:
                return self._process_compare(cast(ast.Compare, node))
            case ast.BinOp:
                return self._process_binop(cast(ast.BinOp, node))
            case ast.UnaryOp:
                return self._process_unaryop(cast(ast.UnaryOp, node))
            #case ast.List:
            #    return [self._process_node(elt) for elt in node.elts]
            #case ast.Tuple:
            #    return tuple(self._process_node(elt) for elt in node.elts)
            case _:
                raise NotImplementedError(f"Node type {node_type.__name__} not supported")

    def _process_call(self, node: ast.Call) -> Any:
        """Process function calls (e.g., pl.col(), methods)."""

        func_type = type(node.func)
        match func_type:
            case ast.Attribute:
                # TODO: rework this
                # Method call like pl.col('x').sum()
                #obj = self._process_node(node.func.value)
                #method = node.func.attr
                #args = [self._process_node(arg) for arg in node.args]
                #kwargs = {kw.arg: self._process_node(kw.value) for kw in node.keywords}
                
                #return self._apply_method(obj, method, args, kwargs)
                raise NotImplementedError(f"Node.func type ast.Attribute not supported")
            
            case ast.Name:
                # Function call like col('x')
                func_name = cast(ast.Name, node.func).id
                args = [self._process_node(arg) for arg in node.args]
                
                if func_name == 'col':
                    return ExpressionNode.column_ref(args[0]) if args else None
            case _:
                return None

    def _process_attribute(self, node: ast.Attribute) -> Any:
        """Process attribute access like pl.col or obj.attr."""
        #TODO rework this
        obj = self._process_node(node.value)
        attr = node.attr

        # Handle pl.col pattern
        #if obj == 'pl' and attr == 'col':
        #    return 'col'
        
        #return f"{obj}.{attr}"
        raise NotImplementedError(f"{obj}.{attr}: Node type ast.Attribute not supported")

    def _process_compare(self, node: ast.Compare) -> Any:
        """Process comparison operations and apply filters."""
        
        left = self._process_node(node.left)
        
        # Handle multiple comparisons
        for op, comparator in zip(node.ops, node.comparators):
            right = self._process_node(comparator)
            expr_node = None
            op_type = type(op)

            match op_type:
                case ast.Eq:
                    expr_node = ExpressionNode.compose(left, OperationType.EQ, right)
                case ast.NotEq:
                    expr_node = ExpressionNode.compose(left, OperationType.NE, right)
                case ast.Lt:
                    expr_node = ExpressionNode.compose(left, OperationType.LT, right)
                case ast.LtE:
                    expr_node = ExpressionNode.compose(left, OperationType.LE, right)
                case ast.Gt:
                    expr_node = ExpressionNode.compose(left, OperationType.GT, right)
                case ast.GtE:
                    expr_node = ExpressionNode.compose(left, OperationType.GE, right)
                case ast.In:
                    expr_node = ExpressionNode.compose(left, OperationType.ISIN, right)
                case ast.NotIn:
                    expr_node = ExpressionNode.compose(left, OperationType.ISNOTIN, right)
                case _:
                    raise NotImplementedError(f"Operator {op_type} not supported")
            left = expr_node
        
        return expr_node

    def _process_binop(self, node: ast.BinOp) -> Any:
        """Process binary operations."""
        
        left = self._process_node(node.left)
        
        right = self._process_node(node.right)
        op_type = type(node.op)

        match op_type:
            case ast.Add:
                return ExpressionNode.compose(left, OperationType.ADD, right)
            case ast.Sub:
                return ExpressionNode.compose(left, OperationType.SUB, right)
            case ast.Mult:
                return ExpressionNode.compose(left, OperationType.MUL, right)
            case ast.Div:
                return ExpressionNode.compose(left, OperationType.DIV, right)
            # TODO: operator & and | can be used for boolean operations between
            # filters, or bitwise operations between integers values. The current
            # implementation supports boolean operations only (which should be
            # the most common case)
            case ast.BitOr:
                return ExpressionNode.compose(left, OperationType.OR, right)
            case ast.BitAnd:
                return ExpressionNode.compose(left, OperationType.AND, right)
            case _:
                raise NotImplementedError(f"Operator {op_type} not supported")
    
    def _process_unaryop(self, node: ast.UnaryOp) -> Any:
        """Process unary operations."""
        
        operand = self._process_node(node.operand)
        op_type = type(node.op)

        match op_type:
            case ast.Invert:
                return ExpressionNode.compose(operand, OperationType.NOT, None)
            case ast.USub:
                return ExpressionNode.compose(operand, OperationType.NEG, None)
            case _:
                raise NotImplementedError(f"Operator {op_type} not supported")
            
        return None

def parse_schema(
    lib: Library,
    symbol: str,
    as_of: int | str | dt.datetime | None = None
) -> pl.Schema:
    arrow_df = lib.read(symbol, as_of=as_of, output_format=OutputFormat.EXPERIMENTAL_ARROW, row_range=((0,1))).data
    return pl.Schema(arrow_df.schema) 

def scan_arcticdb(
    uri: str,
    lib_name: str,
    symbol: str,
    as_of: int | str | dt.datetime | None = None
) -> pl.LazyFrame:

    ac = Arctic(uri)
    lib = ac.get_library(lib_name)

    schema = parse_schema(lib, symbol, as_of)

    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None
    ) -> Iterator[pl.DataFrame]:

        qb = None
        if predicate is not None:
            tl = PolarsToArcticDBTranslator()
            qb = QueryBuilder()
            qb = tl.translate(predicate, qb)
        
        # TODO: convert predicate to QueryBuilder and pass it to read
        lazy_df = lib.read(symbol, as_of = as_of, columns = with_columns, query_builder = qb, lazy = True, output_format=OutputFormat.EXPERIMENTAL_ARROW)

        if batch_size is None:
            batch_size = 1000

        if n_rows is not None:
            batch_size = min(batch_size, n_rows)
        
        read_idx = 0
        while n_rows is None or n_rows > 0:
            lazy_df_slice = lazy_df.row_range((read_idx, read_idx + batch_size))
            read_idx += batch_size
            arrow_df = lazy_df_slice.collect().data
            if n_rows is not None:
                n_rows -= arrow_df.num_rows
            elif arrow_df.num_rows < batch_size:
                n_rows = 0

            yield cast(pl.DataFrame, pl.from_arrow(arrow_df))

    return pl.io.plugins.register_io_source(io_source=source_generator, schema = schema)    
