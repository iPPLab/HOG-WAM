import json

from tools.summarize_eval_results import main


def test_summary_counts_task_results(tmp_path, monkeypatch, capsys):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "0.json").write_text(json.dumps({"task": "task", "seed": 0, "success": True}), encoding="utf-8")
    (task_dir / "1.json").write_text(json.dumps({"task": "task", "seed": 1, "success": False}), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["summarize_eval_results.py", str(tmp_path), "--episodes-per-task", "2"])
    main()
    result = json.loads(capsys.readouterr().out)
    assert result["episodes"] == 2
    assert result["successes"] == 1
    assert result["success_rate_percent"] == 50.0
    assert result["tasks"][0]["complete"] is True
