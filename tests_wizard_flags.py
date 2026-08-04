"""The interactive path must accept every flag the parser declares."""

from subs2dub import cli, wizard


def test_wizard_namespace_matches_parser():
    ns = wizard._to_args(
        "v.mkv", "o.mkv", "styletts2", "uk", "en", "claude",
        True, True, None, -16.0,
    )
    declared = set(vars(cli.defaults_for_build())) - {"func", "cmd"}
    supplied = set(vars(ns))
    assert not declared - supplied, f"wizard missing: {sorted(declared - supplied)}"


if __name__ == "__main__":
    test_wizard_namespace_matches_parser()
    print("wizard namespace matches the parser")
