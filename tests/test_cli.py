import json
from pathlib import Path

from mhcmtn.__main__ import main


def test_cli_creates_and_verifies_an_actual_checkpoint(tmp_path, capsys):
    case = Path(__file__).parents[1] / "examples/synthetic_day.json"
    main(["optimize", "--case", str(case), "--output", str(tmp_path), "--epochs", "2", "--no-plot"])
    result = json.loads((tmp_path / "result.json").read_text())
    capsys.readouterr()
    main(["check", "--checkpoint", str(tmp_path / "checkpoint.pt")])
    checked = json.loads(capsys.readouterr().out)
    assert result["objective"] == checked["objective"]
    assert (tmp_path / "schedule.csv").exists() and (tmp_path / "history.csv").exists()
