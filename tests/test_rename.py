"""The name before 0.5.0 keeps working for one release."""
import os
import subprocess
import sys
from pathlib import Path

from ditto_code import claude, cli, ladder
from ditto_code.ladder import Measurements
from ditto_code.proxy import Config

SRC = str(Path(__file__).resolve().parents[1] / "src")


def test_models_named_with_the_old_prefix_are_still_rungs(tmp_path):
    tags = [{"name": "ditto-code/a:64k", "digest": "1", "size": 10}, {"name": "understudy/b:64k", "digest": "2", "size": 20},
            {"name": "other/c", "digest": "3", "size": 5}]
    cfg = Config()
    names = [r.name for r in ladder.rungs(tags, cfg.prefixes, Measurements(tmp_path / "m.json"))]
    assert sorted(names) == ["ditto-code/a:64k", "understudy/b:64k"]
    assert Config(prefix="mine/").prefixes == ("mine/", "understudy/")
    assert Config(legacy_prefixes=()).prefixes == ("ditto-code/",)


def test_the_old_import_and_module_reach_the_new_code():
    env = dict(os.environ, PYTHONPATH=SRC)
    out = subprocess.run([sys.executable, "-c", "from understudy import claude, proxy; import understudy.cli as c; "
                          "import ditto_code.claude as n; print(claude is n, c.main.__module__)"],
                         capture_output=True, text=True, env=env)
    assert out.stdout.split() == ["True", "ditto_code.cli"], out.stderr
    run = subprocess.run([sys.executable, "-m", "understudy.cli", "--help"], capture_output=True, text=True, env=env)
    assert run.returncode == 0 and "ditto-code" in run.stdout


def test_the_old_command_says_it_was_renamed_and_runs(capsys, monkeypatch):
    monkeypatch.setattr(cli, "main", lambda argv=None: 7)
    assert cli.old_name([]) == 7
    assert "now ditto-code" in capsys.readouterr().err


def test_the_old_environment_names_are_read_when_the_new_ones_are_not_set(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(claude, "default_path", lambda home=None: tmp_path / "s.json")
    monkeypatch.setattr(claude, "write_settings", lambda path, url, *a, keep_mcp=(), **k: seen.update(url=url, keep=keep_mcp))
    monkeypatch.setattr(claude, "link_skill", lambda home=None: "")
    for name in ("DITTO_CODE_URL", "DITTO_CODE_KEEP_MCP"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("UNDERSTUDY_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("UNDERSTUDY_KEEP_MCP", "old")
    cli.main(["setup-claude"])
    assert seen == {"url": "http://127.0.0.1:9", "keep": ["old"]}
    monkeypatch.setenv("DITTO_CODE_URL", "http://127.0.0.1:8")
    monkeypatch.setenv("DITTO_CODE_KEEP_MCP", "new")
    cli.main(["setup-claude"])
    assert seen == {"url": "http://127.0.0.1:8", "keep": ["new"]}


def test_setup_removes_the_skill_link_the_old_name_made(tmp_path):
    skills = tmp_path / ".claude" / "skills"
    skills.mkdir(parents=True)
    (skills / "understudy").symlink_to(tmp_path / "pkg" / "skills" / "understudy")
    (skills / "kept").symlink_to(tmp_path / "elsewhere")
    claude.link_skill(tmp_path)
    assert not (skills / "understudy").is_symlink() and (skills / "kept").is_symlink()
    assert os.readlink(skills / "ditto-code").endswith(os.path.join("skills", "ditto-code"))
