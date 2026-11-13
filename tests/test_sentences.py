import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/tone"))
from sentences import keep_sentence, split_sentences  # noqa: E402


def test_split_protects_abbreviations_and_decimals():
    t = "The U.S. economy grew 2.5 per cent. Inflation eased to 3.1 per cent. Dr. Patra noted risks."
    assert split_sentences(t) == ["The U.S. economy grew 2.5 per cent.", "Inflation eased to 3.1 per cent.",
                                  "Dr. Patra noted risks."]


def test_split_keeps_initials():
    t = "Jerome H. Powell chaired the meeting of the Committee today. Rates were unchanged at this meeting."
    assert len(split_sentences(t)) == 2


def test_filters_drop_name_lists_and_references():
    assert not keep_sentence("Roberto Perli, Manager, System Open Market Account")
    assert not keep_sentence("Ref: Cogley, T., Primiceri, G. E., & Sargent, T. J. (2010).")
    assert keep_sentence("Inflation has made progress toward the Committee's 2 percent objective but remains elevated.")
