import json

import pytest

from ticketmodel import cli, site
from ticketmodel.config import CAPACITY, Paths


@pytest.fixture
def trained(fixture_root):
    paths = Paths(fixture_root)
    cli.cmd_train(paths)
    cli.cmd_predict(paths)
    paths.logos_dir.mkdir()
    (paths.logos_dir / "msu.png").write_bytes(b"msu")
    return paths


def by_game(data):
    return {(g["season"], g["opponent"]): g for g in data["games"]}


def test_site_data_assigns_status_tier_and_logo(trained):
    data = site.site_data(trained, today="2024-10-20")
    by = by_game(data)
    assert len(by) == 12 and data["seasons"] == [2023, 2024]
    a = by[(2023, "Rival A")]
    assert a["status"] == "played" and a["tier"] == "tier2" and a["opp_id"] == 333 and a["logo"] == "333.png"
    assert a["slug"] == "2023-rival-a"
    assert by[(2024, "Mid Major")]["tier"] == "tier1"                       # played, no price
    assert by[(2024, "Rival C")]["status"] == "upcoming" and by[(2024, "Rival C")]["tier"] == "tier2"
    assert by[(2024, "Rival D")]["status"] == "upcoming" and by[(2024, "Rival D")]["tier"] == "tier1"
    assert data["next_game"] == "2024-rival-c"


def test_site_data_played_games_use_leave_one_out_forecast(trained):
    data = site.site_data(trained, today="2024-10-20")
    g = by_game(data)[(2023, "Rival A")]
    row = next(r for r in json.loads(trained.train_summary.read_text())["per_game"]
               if r["season"] == 2023 and r["opponent"] == "Rival A")
    assert g["forecast"]["pred"] == row["tier2_loo"] and g["forecast"]["lo"] == row["tier2_lo"]
    assert g["actual"] == 60000 and g["error"] == g["forecast"]["pred"] - 60000
    assert g["inside"] == (row["tier2_lo"] <= 60000 <= row["tier2_hi"])
    assert data["track"]["n"] == 10 and 0 <= data["track"]["inside_rate"] <= 1


def test_site_data_breakdown_sums_to_the_unclipped_forecast(trained):
    data = site.site_data(trained, today="2024-10-20")
    for g in data["games"]:
        if g["forecast"] is None:
            continue
        b = g["breakdown"]
        assert b["baseline"] + sum(t["contribution"] for t in b["terms"]) == pytest.approx(b["total"], abs=0.5)
        assert min(max(b["total"], 0), CAPACITY) == pytest.approx(g["forecast"]["pred"], abs=1)
        assert 0 <= g["forecast"]["p_sellout"] <= 1
        assert [t["feature"] for t in b["terms"]] == data["models"][g["tier"]]["features"]
    up = by_game(data)[(2024, "Rival C")]
    assert up["forecast"]["pred"] == pytest.approx(up["breakdown"]["total"], abs=1) or up["forecast"]["pred"] == CAPACITY


def test_site_data_models_carry_labels_means_and_swing(trained):
    data = site.site_data(trained, today="2024-10-20")
    t2 = data["models"]["tier2"]
    assert "rel_log_price" in t2["features"] or "log_getin" in t2["features"]
    for f in t2["features"]:
        assert t2["labels"][f] and f in t2["means"] and f in t2["swing"]
    assert t2["baseline"] == pytest.approx(t2["intercept"] + sum(t2["coef"][f] * t2["means"][f] for f in t2["features"]))
    assert data["metrics"]["Tier 2 (priced rows)"]["rmse"] > 0


def test_build_site_writes_pages_logos_and_seo_files(trained, monkeypatch):
    monkeypatch.setenv("SITE_URL", "https://example.netlify.app")
    calls = []

    def http(url):
        calls.append(url)
        return 200, b"png"

    site.build_site(trained, http=http, today="2024-10-20")
    out = trained.site_dir
    for name in ["index.html", "track-record.html", "model.html", "site.css", "site.js", "data.json",
                 "robots.txt", "sitemap.xml", "games/2024-rival-c.html", "logos/msu.png", "logos/333.png"]:
        assert (out / name).exists(), name
    assert len(calls) == 6                                   # six distinct home opponents, MSU excluded
    html = (out / "index.html").read_text()
    assert "Rival C" in html and "logos/99.png" in html and "logos/msu.png" in html
    assert html.count("<h1") == 1 and "<title>" in html and 'rel="canonical"' in html
    assert "ClaudeBot" in (out / "robots.txt").read_text()
    assert "games/2023-rival-a.html" in (out / "sitemap.xml").read_text()
    assert json.loads((out / "data.json").read_text())["capacity"] == CAPACITY


def test_build_site_falls_back_when_a_logo_cannot_be_fetched(trained, capsys):
    site.build_site(trained, http=lambda url: (404, b""), today="2024-10-20")
    html = (trained.site_dir / "games" / "2024-rival-c.html").read_text()
    assert "logo-fallback" in html and "logos/99.png" not in html
    assert "logo" in capsys.readouterr().out


def test_site_command_from_cli(trained, monkeypatch):
    monkeypatch.setattr(site.logos, "default_http", lambda url: (200, b"png"))
    assert cli.main(["site", "--root", str(trained.root)]) == 0
    assert (trained.site_dir / "index.html").exists()


def test_site_has_no_links_back_to_the_author_when_no_site_url(trained, monkeypatch):
    monkeypatch.delenv("SITE_URL", raising=False)
    site.build_site(trained, http=lambda url: (200, b"png"), today="2024-10-20")
    out = trained.site_dir
    for page in list(out.rglob("*.html")) + [out / "data.json"]:
        text = page.read_text()
        assert "github.com" not in text and "gardnerthornhill" not in text, page
    html = (out / "index.html").read_text()
    assert 'rel="canonical"' not in html and "og:url" not in html
    assert "Sitemap:" not in (out / "robots.txt").read_text()
    assert not (out / "sitemap.xml").exists()


def test_site_url_from_environment_drives_canonical_and_sitemap(trained, monkeypatch):
    monkeypatch.setenv("SITE_URL", "https://example.netlify.app/")
    site.build_site(trained, http=lambda url: (200, b"png"), today="2024-10-20")
    out = trained.site_dir
    html = (out / "games" / "2024-rival-c.html").read_text()
    assert '<link rel="canonical" href="https://example.netlify.app/games/2024-rival-c.html">' in html
    assert "https://example.netlify.app/sitemap.xml" in (out / "robots.txt").read_text()
    assert "https://example.netlify.app/track-record.html" in (out / "sitemap.xml").read_text()
