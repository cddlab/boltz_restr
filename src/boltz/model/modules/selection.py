from typing import List, Union, Dict, Any, Protocol, runtime_checkable

@runtime_checkable
class SelectionNode(Protocol):
    def eval(self, mol: Dict[str, Union[str, int]]) -> bool:
        ...

    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.__dict__ == other.__dict__


class Chain(SelectionNode):
    def __init__(self, names: List[str]):
        self.names = names
    def eval(self, mol: Dict[str, Union[str, int]]) -> bool:
        chain = mol.get("chain")
        return isinstance(chain, str) and chain in self.names

# class ResName(SelectionNode):
#     def __init__(self, names: List[str]):
#         self.names = names
#     def eval(self, mol: Dict[str, Union[str, int]]) -> bool:
#         resname = mol.get("resname")
#         return isinstance(resname, str) and resname in self.names

class ResId(SelectionNode):
    def __init__(self, ids: List[int]):
        self.ids = ids
    def eval(self, mol: Dict[str, Union[str, int]]) -> bool:
        resid = mol.get("resid")
        return isinstance(resid, int) and resid in self.ids

# class Name(SelectionNode):
#     def __init__(self, names: List[str]):
#         self.names = names
#     def eval(self, mol: Dict[str, Union[str, int]]) -> bool:
#         name = mol.get("name")
#         return isinstance(name, str) and name in self.names

class Index(SelectionNode):
    def __init__(self, indices: List[int]):
        self.indices = indices
    def eval(self, mol: Dict[str, Union[str, int]]) -> bool:
        index = mol.get("index")
        return isinstance(index, int) and index in self.indices

class Not(SelectionNode):
    def __init__(self, selection: SelectionNode):
        self.selection = selection
    def eval(self, mol: Dict[str, Union[str, int]]) -> bool:
        return not self.selection.eval(mol)

class And(SelectionNode):
    def __init__(self, selections: List[SelectionNode]):
        self.selections = selections
    def eval(self, mol: Dict[str, Union[str, int]]) -> bool:
        if not self.selections: return True
        return all(s.eval(mol) for s in self.selections)

class Or(SelectionNode):
    def __init__(self, selections: List[SelectionNode]):
        self.selections = selections
    def eval(self, mol: Dict[str, Union[str, int]]) -> bool:
        if not self.selections: return False
        return any(s.eval(mol) for s in self.selections)

class Bracket(SelectionNode):
    def __init__(self, selection: SelectionNode):
        self.selection = selection
    def eval(self, mol: Dict[str, Union[str, int]]) -> bool:
        return self.selection.eval(mol)

# --- Parser Error ---
class ParseError(ValueError):
    pass

# --- Parser Class ---
# RESERVED_KEYWORDS = {"and", "or", "not", "to", "resname", "resid", "name", "index", "chain"}
RESERVED_KEYWORDS = {"and", "or", "not", "to", "resid", "index", "chain"}

class SelectionParser:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def _peek(self) -> Union[str, None]:
        return self.text[self.pos] if self.pos < len(self.text) else None

    def _consume_char(self, char: str):
        if self._peek() == char:
            self.pos += 1
            return char
        raise ParseError(f"Expected '{char}' at position {self.pos}, got '{self._peek()}'")

    def _consume_tag(self, tag: str):
        if self.text.startswith(tag, self.pos):
            if tag.isalpha() and (self.pos + len(tag) < len(self.text) and \
                                  self.text[self.pos + len(tag)].isalnum()):
                pass
            self.pos += len(tag)
            return tag
        raise ParseError(f"Expected '{tag}' at position {self.pos}")

    def _skip_space0(self):
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def _skip_space1(self):
        start_pos = self.pos
        self._skip_space0()
        if self.pos == start_pos:
            raise ParseError(f"Expected one or more spaces at position {self.pos}")

    def _parse_alphanumeric1(self) -> str:
        start_pos = self.pos
        if self.pos < len(self.text) and self.text[self.pos].isalnum():
            self.pos += 1
            while self.pos < len(self.text) and self.text[self.pos].isalnum():
                self.pos += 1
            return self.text[start_pos:self.pos]
        raise ParseError(f"Expected alphanumeric characters at position {self.pos}")

    def _parse_digit1(self) -> str:
        start_pos = self.pos
        if self.pos < len(self.text) and self.text[self.pos].isdigit():
            self.pos += 1
            while self.pos < len(self.text) and self.text[self.pos].isdigit():
                self.pos += 1
            return self.text[start_pos:self.pos]
        raise ParseError(f"Expected digits at position {self.pos}")

    def _parse_usize(self) -> int:
        s = self._parse_digit1()
        try:
            return int(s)
        except ValueError:
            raise ParseError(f"Invalid unsigned integer: {s}")

    def _parse_identifier(self) -> str:
        identifier = self._parse_alphanumeric1()
        # if identifier in RESERVED_KEYWORDS - {"resname", "resid", "name", "index"}:
        if identifier in RESERVED_KEYWORDS - {"resid", "index"}:
            if identifier in {"and", "or", "not", "to"}:
                raise ParseError(f"Identifier cannot be a reserved keyword: '{identifier}' at position {self.pos - len(identifier)}")
        return identifier

    def _parse_list_of_identifiers(self) -> List[str]:
        identifiers = [self._parse_identifier()]
        while True:
            saved_pos = self.pos
            try:
                self._skip_space1()
                identifiers.append(self._parse_identifier())
            except ParseError:
                self.pos = saved_pos
                break
        return identifiers

    def _parse_numbers(self) -> List[int]:
        first = self._parse_usize()

        saved_pos_for_to = self.pos
        try:
            self._skip_space1()
            self._consume_tag("to")
            self._skip_space1()
            last = self._parse_usize()
            if last < first:
                raise ParseError(f"Range end {last} is less than start {first}")
            return list(range(first, last + 1))
        except ParseError:
            self.pos = saved_pos_for_to
            numbers = [first]
            while True:
                saved_pos_loop = self.pos
                try:
                    self._skip_space1()
                    numbers.append(self._parse_usize())
                except ParseError:
                    self.pos = saved_pos_loop
                    break
            return numbers

    # def _parse_resname(self) -> SelectionNode:
    #     self._consume_tag("resname")
    #     self._skip_space1()
    #     return ResName(self._parse_list_of_identifiers())

    # def _parse_name(self) -> SelectionNode:
    #     self._consume_tag("name")
    #     self._skip_space1()
    #     return Name(self._parse_list_of_identifiers())

    def _parse_resid(self) -> SelectionNode:
        self._consume_tag("resid")
        self._skip_space1()
        return ResId(self._parse_numbers())

    def _parse_index(self) -> SelectionNode:
        self._consume_tag("index")
        self._skip_space1()
        return Index(self._parse_numbers())

    def _parse_chain(self) -> SelectionNode:
        self._consume_tag("chain")
        self._skip_space1()
        return Chain(self._parse_list_of_identifiers())

    # --- Grammar hierarchy ---
    def _parse_atom(self) -> SelectionNode:
        atom_parsers = [
            # self._parse_chain, self._parse_resname, self._parse_resid, self._parse_index, self._parse_name,
            self._parse_chain, self._parse_resid, self._parse_index,
        ]
        for parser_func in atom_parsers:
            saved_pos = self.pos
            try:
                return parser_func()
            except ParseError:
                self.pos = saved_pos
        raise ParseError(f"Expected an atomic selection (e.g., 'all', 'index 1') at position {self.pos}")

    def _parse_bracket(self) -> SelectionNode:
        self._consume_char('(')
        expr = self.parse_expr()
        self._consume_char(')')
        return Bracket(expr)

    def _parse_primary(self) -> SelectionNode:
        self._skip_space0()
        saved_pos = self.pos
        try:
            return self._parse_bracket()
        except ParseError:
            self.pos = saved_pos
            try:
                return self._parse_atom()
            except ParseError as e_atom:
                if self.text[saved_pos:].startswith("("):
                    raise ParseError(f"Syntax error in parenthesized expression or mismatched parentheses near position {saved_pos}") from e_atom
                raise

    def _parse_not(self) -> SelectionNode:
        num_nots = 0
        while True:
            saved_pos = self.pos
            self._skip_space0()
            try:
                self._consume_tag("not")
                num_nots += 1
            except ParseError:
                self.pos = saved_pos
                break

        selection = self._parse_primary()

        for _ in range(num_nots):
            selection = Not(selection)
        return selection

    def _parse_and(self) -> SelectionNode:
        operands = [self._parse_not()]
        while True:
            saved_pos = self.pos
            try:
                self._skip_space1()
                self._consume_tag("and")
                self._skip_space1()
                operands.append(self._parse_not())
            except ParseError:
                self.pos = saved_pos
                break
        return operands[0] if len(operands) == 1 else And(operands)

    def _parse_or(self) -> SelectionNode:
        operands = [self._parse_and()]
        while True:
            saved_pos = self.pos
            try:
                self._skip_space1()
                self._consume_tag("or")
                self._skip_space1()
                operands.append(self._parse_and())
            except ParseError:
                self.pos = saved_pos
                break
        return operands[0] if len(operands) == 1 else Or(operands)

    def parse_expr(self) -> SelectionNode:
        expr = self._parse_or()
        self._skip_space0()
        return expr

    def parse(self) -> SelectionNode:
        parsed_node = self.parse_expr()
        if self.pos < len(self.text):
            raise ParseError(f"Unexpected trailing characters: '{self.text[self.pos:]}' at position {self.pos}")
        return parsed_node


# --- Public API as per Rust code ---
def parse_selection(selection_string: str) -> Union[SelectionNode, str]:
    try:
        parser = SelectionParser(selection_string)
        return parser.parse()
    except ParseError as e:
        return str(e)

# --- AtomSelector Class (User's starting point) ---
class AtomSelector:
    def __init__(self, selection_string: str) -> None:
        self.selection_string = selection_string
        self.parsed_selection: Union[SelectionNode, None] = None
        self._error: Union[str, None] = None

        result = parse_selection(selection_string)
        if isinstance(result, SelectionNode):
            self.parsed_selection = result
        else:
            self._error = result
            raise ValueError(f"Failed to parse selection string: {self._error}")


    def eval(self, mol: Dict[str, Union[str, int]]) -> bool:
        """
        mol: Dict[str, str | int]
        e.g.
        # mol = { "chain": "A", "resname": "ALA", "resid": 1, "name": "CA", "index": 0 }
        mol = { "chain": "A", "resid": 1, "index": 0 }
        """
        if self._error:
            print(f"Cannot evaluate: parsing failed with error: {self._error}")
            return False
        if self.parsed_selection is None:
            raise RuntimeError("Selection was not successfully parsed.")

        return self.parsed_selection.eval(mol)


if __name__ == "__main__":

    molecules = [
        # { "chain": "A", "resname": "ALA", "resid": 1, "name": "CA", "index": 0 },
        # { "chain": "B", "resname": "ALA", "resid": 1, "name": "CA", "index": 0 },
        # { "chain": "A", "resname": "ALA", "resid": 11, "name": "CA", "index": 0 },
        { "chain": "A", "resid": 1, "index": 0 },
        { "chain": "B", "resid": 1, "index": 0 },
        { "chain": "A", "resid": 11, "index": 0 },
    ]

    atom_selection = "(not chain A) and resid 1 to 11"
    print(f"{atom_selection=}")
    atom_selector = AtomSelector(atom_selection)

    for mol in molecules:
        print(mol, atom_selector.eval(mol))

