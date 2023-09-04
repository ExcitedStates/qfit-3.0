import operator

import numpy as np

import pyparsing as pp


class _Selector:

    """
    Parse and evaluate selection syntax.

    Based on PyMol's selection syntax.
    """

    def __init__(self):
        self.structure = None

        # Name selectors
        kw_chain = pp.Keyword("chain")
        kw_resn = pp.Keyword("resn")
        kw_altloc = pp.Keyword("altloc")
        kw_name = pp.Keyword("name")
        kw_icode = pp.Keyword("icode")
        kw_resseq = pp.Keyword("resseq")

        # Resi is special cause it can take a range
        kw_resi = pp.Keyword("resi")

        # Number selectors
        kw_b = pp.Keyword("b")
        kw_q = pp.Keyword("q")

        identifyer_selectors = (
            kw_chain | kw_resn | kw_altloc | kw_name | kw_icode | kw_resseq
        )
        numeric_selectors = kw_b | kw_q

        # operators
        kw_or = pp.Keyword("or")
        kw_and = pp.Keyword("and")
        kw_and_or = kw_or | kw_and
        kw_not = pp.Keyword("not")

        plus = pp.Literal("+").suppress()
        minus = pp.Literal("-")
        plus_minus = plus | minus
        lpar = pp.Literal("(").suppress()
        rpar = pp.Literal(")").suppress()
        point = pp.Literal(".")
        operators = (
            pp.Literal(">=")
            | pp.Literal("!=")
            | pp.Literal("<=")
            | pp.Literal(">")
            | pp.Literal("<")
            | pp.Literal("==")
        )
        empty = pp.Literal('""') | pp.Literal("''")

        identifyer = pp.Word(pp.alphanums + "'" + ".") | empty
        number = pp.Word(pp.nums)
        integer = pp.Combine(pp.Optional(plus_minus) + number)
        floatnumber = pp.Combine(
            integer + pp.Optional(point + pp.Optional(number))
        ).setParseAction(self._push_first)

        resi_number = integer
        arange = resi_number + pp.Optional(minus + resi_number)

        identifyers = identifyer + pp.ZeroOrMore(plus + identifyer)
        identifyer_selection = identifyer_selectors + identifyers.setParseAction(
            self._push
        )
        resi_selection = kw_resi + (
            arange + pp.ZeroOrMore(plus + arange)
        ).setParseAction(self._push)
        numeric_selection = numeric_selectors + (
            operators + floatnumber
        ).setParseAction(self._push_first)

        expression = pp.Forward()
        selections = identifyer_selection | resi_selection | numeric_selection
        atom = selections.setParseAction(self._push_first) | lpar + expression + rpar
        factor = (kw_not + atom).setParseAction(self._push_first) | atom
        # PyMol doesnt specify precedence of "or" and "and"
        expression << factor + pp.ZeroOrMore(
            (kw_and_or + factor).setParseAction(self._push_first)
        )
        self.expression = expression

    def set_structure(self, structure):
        if id(self.structure) == id(structure):
            return
        self.structure = structure
        self.curr_sel = self.structure.selection
        if self.curr_sel is None:
            self.curr_sel = np.arange(self.structure.total_length, dtype=np.uintp)

    def _push_first(self, strg, loc, toks):
        self.expr_stack.append(toks[0])

    def _push(self, strg, loc, toks):
        self.expr_stack.append(toks[::-1])

    def _evaluate_stack(self, s):
        token = s.pop()
        if token == "or":
            sel1 = self._evaluate_stack(s)
            sel2 = self._evaluate_stack(s)
            return np.union1d(sel1, sel2)
        elif token == "and":
            sel1 = self._evaluate_stack(s)
            sel2 = self._evaluate_stack(s)
            return np.intersect1d(sel1, sel2, True)
        elif token == "not":
            return np.setdiff1d(self.curr_sel, self._evaluate_stack(s), True)
        elif token == "resi":
            elements = s.pop()
            resi = self.structure.get_array(token)
            selections = []
            while elements:
                if len(elements) > 1 and elements[-2] == "-":
                    start = int(elements.pop())
                    elements.pop()
                    end = int(elements.pop())
                    mask = operator.ge(resi, start) & operator.le(resi, end)
                else:
                    value = int(elements.pop())
                    mask = resi == value
                selections.append(self.curr_sel[mask])
            selection = np.unique(np.concatenate(selections))
            return selection
        elif token in ("resn", "chain", "name", "altloc", "icode"):
            data = self.structure.get_array(token)
            picks = set(s.pop())
            selections = []
            for value in picks:
                if value in ('""', "''"):
                    value = ""
                mask = data == value
                selections.append(self.curr_sel[mask])
            selection = np.concatenate(selections)
            return selection
        elif token == "resseq":
            picks = set(s.pop())
            selections = []
            resi_data = self.structure.get_array("resi")
            icode_data = self.structure.get_array("icode")
            for value in picks:
                try:
                    resi, icode = value.split(".")
                except ValueError:
                    resi = value
                    icode = ""
                resi = int(resi)
                mask = resi_data == resi
                if icode:
                    mask &= icode_data == icode
                selections.append(self.curr_sel[mask])
            selection = np.concatenate(selections)
            return selection
        elif token in ("q", "b"):
            data = self.structure.get_array(token)
            operator_str = s.pop()
            float_number = float(s.pop())
            loper = self._get_operator(operator_str)
            mask = loper(data, float_number)
            selection = self.curr_sel[mask]
            return selection

    @staticmethod
    def _get_operator(loperator):
        if loperator in ("==", "!="):
            oper = operator.eq
        elif loperator == "<":
            oper = operator.lt
        elif loperator == ">":
            oper = operator.gt
        elif loperator == ">=":
            oper = operator.ge
        elif loperator == "<=":
            oper = operator.le
        else:
            raise ValueError("Logic operator not recognized.")
        return oper

    def __call__(self, string, parse_all=True):
        self.expr_stack = []
        self.expression.parseString(string, parse_all)
        selection = self._evaluate_stack(self.expr_stack)
        return selection
