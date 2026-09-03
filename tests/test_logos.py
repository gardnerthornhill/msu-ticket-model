from ticketmodel import logos


def test_ensure_logos_downloads_only_missing_ids(tmp_path):
    calls = []

    def http(url):
        calls.append(url)
        return 200, b"PNGDATA"

    (tmp_path / "333.png").write_bytes(b"already here")
    failed = logos.ensure_logos([333, 2], tmp_path, http=http)
    assert calls == ["https://a.espncdn.com/i/teamlogos/ncaa/500/2.png"]
    assert (tmp_path / "2.png").read_bytes() == b"PNGDATA"
    assert (tmp_path / "333.png").read_bytes() == b"already here"
    assert failed == []


def test_ensure_logos_reports_failures_without_raising(tmp_path):
    def http(url):
        return 404, b""

    failed = logos.ensure_logos([5], tmp_path, http=http)
    assert failed == [5]
    assert not (tmp_path / "5.png").exists()


def test_logo_path_prefers_the_msu_mark_for_the_home_team(tmp_path):
    assert logos.logo_file(344) == "msu.png"
    assert logos.logo_file(333) == "333.png"
