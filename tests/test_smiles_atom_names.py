# ruff: noqa: INP001

from boltz.data.parse.schema import _smiles_atom_names

_MAX_ATOM_NAME_LENGTH = 4


def test_smiles_atom_names_preserve_existing_scheme_when_it_fits() -> None:
    """Short canonical names retain the historical global-rank scheme."""
    names = _smiles_atom_names(["C", "Cl", "O"], [2, 0, 1])

    assert names == ["C3", "CL1", "O2"]


def test_smiles_atom_names_reindex_by_element_on_overflow() -> None:
    """A CL100 overflow switches to unique element-local names."""
    symbols = ["C"] * 99 + ["Cl"] + ["H"] * 80
    names = _smiles_atom_names(symbols, list(range(len(symbols))))

    assert names[99] == "CL1"
    assert len(names) == len(set(names))
    assert all(len(name) <= _MAX_ATOM_NAME_LENGTH for name in names)
