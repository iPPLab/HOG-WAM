"""Orbax list-index key spelling, shared by every module that restores a tree.

flax 0.10's ``to_pure_dict`` spelled list indices as the strings "0", "1", ... and that
is how the published Orbax checkpoint is written (its ``_METADATA``
records ``key_type: 2``, i.e. string, for each ``blocks`` index). flax 0.12 spells them
as ints. Restoring one against the other is an orbax structure mismatch on a tree that
is in fact identical, so every loader has to translate.

This lived twice — once in ``vae.py`` as ``_stringify_keys`` / ``_restore_key_types``,
once in ``transformer.py`` as the public pair — with a note in each saying "hoist this
if a third consumer shows up". M2's ActionDiT / MoT / official-checkpoint loaders are
that third consumer, so it is hoisted. Both old modules re-export these names, so
nothing that imported them from there breaks.
"""

__all__ = ["stringify_tree_keys", "restore_tree_key_types"]


def stringify_tree_keys(tree):
    """int dict keys -> str, recursively. Non-dict leaves pass through untouched."""
    if isinstance(tree, dict):
        return {str(key): stringify_tree_keys(value) for key, value in tree.items()}
    return tree


def restore_tree_key_types(tree, like):
    """Inverse of :func:`stringify_tree_keys`, taking the key types from ``like``.

    ``like`` drives the walk, so a key present in ``tree`` but absent from ``like`` is
    dropped and a key missing from ``tree`` raises ``KeyError`` — which is what the
    callers want: the restore target is the authority on the shape of the tree.
    """
    if isinstance(tree, dict) and isinstance(like, dict):
        return {key: restore_tree_key_types(tree[str(key)], sub) for key, sub in like.items()}
    return tree
