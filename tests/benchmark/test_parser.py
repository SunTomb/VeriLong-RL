from benchmark.parser.output_parser import parse_model_output


def test_parse_valid_answer_evidence_steps():
    text = """Evidence: E03, E17
Steps:
1. E03 establishes the initial condition.
2. E17 updates the relevant fact.
3. Therefore the answer is Team Delta.
Answer: Team Delta
"""
    parsed = parse_model_output(text, valid_evidence_ids={"E03", "E17", "E21"})
    assert parsed.format_valid is True
    assert parsed.pred_evidence_ids == ["E03", "E17"]
    assert parsed.pred_answer == "Team Delta"
    assert len(parsed.pred_steps) == 3
    assert parsed.error_flags == []


def test_parse_missing_answer_records_error():
    text = "Evidence: E03\nSteps:\n1. E03 supports it."
    parsed = parse_model_output(text, valid_evidence_ids={"E03"})
    assert parsed.format_valid is False
    assert "missing_answer" in parsed.error_flags


def test_parse_invalid_evidence_records_error():
    text = "Evidence: E99\nSteps:\n1. E99 supports it.\nAnswer: X"
    parsed = parse_model_output(text, valid_evidence_ids={"E01"})
    assert parsed.format_valid is False
    assert "invalid_evidence_id:E99" in parsed.error_flags
