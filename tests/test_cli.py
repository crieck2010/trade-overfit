import json

import pytest

from trade_overfit.cli import main, read_returns_csv


def test_read_returns_csv_with_header(tmp_path):
    p = tmp_path / "rets.csv"
    p.write_text("return\n0.01\n-0.02\n0.03\n")
    assert read_returns_csv(str(p)) == pytest.approx([0.01, -0.02, 0.03])


def test_read_returns_csv_empty_raises(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("return\n")
    with pytest.raises(SystemExit):
        read_returns_csv(str(p))


def test_validate_demo_edge_table(capsys):
    assert main(["validate", "--demo", "edge"]) == 0
    out = capsys.readouterr().out
    assert "verdict: PASS" in out


def test_validate_demo_noise_winner_fails(capsys):
    assert main(["validate", "--demo", "noise-winner"]) == 0
    out = capsys.readouterr().out
    assert "verdict: FAIL" in out


def test_validate_json(capsys):
    assert main(["validate", "--demo", "edge", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "PASS"


def test_validate_csv_format(capsys):
    assert main(["validate", "--demo", "edge", "--format", "csv"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("field,value")


def test_dsr_command(capsys):
    assert main(["dsr", "--demo", "noise-winner"]) == 0
    assert "dsr" in capsys.readouterr().out


def test_walkforward_command(capsys):
    assert main(["walk-forward", "--demo", "edge"]) == 0
    assert "n_windows" in capsys.readouterr().out


def test_gates_command_strict(capsys):
    assert main(["gates", "--demo", "edge", "--preset", "strict"]) == 0
    assert "deflated_sharpe" in capsys.readouterr().out


def test_metrics_command(capsys):
    assert main(["metrics", "--demo", "edge"]) == 0
    assert "sharpe" in capsys.readouterr().out


def test_presets_command(capsys):
    assert main(["presets"]) == 0
    out = capsys.readouterr().out
    assert "[standard]" in out and "[strict]" in out and "[lenient]" in out


def test_license_command(capsys):
    assert main(["license"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True


def test_update_check_never_raises(capsys):
    assert main(["update-check"]) == 0  # works offline; reports error field
    payload = json.loads(capsys.readouterr().out)
    assert "current" in payload


def test_validate_from_csv_file(tmp_path, capsys):
    p = tmp_path / "rets.csv"
    p.write_text("\n".join(["0.001"] * 400))
    assert main(["validate", "--csv", str(p), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] in ("PASS", "FAIL")
