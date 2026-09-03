"""Team logos for the site: the MSU mark from logos/, opponents from ESPN's CDN by CFBD team id."""
import urllib.error
import urllib.request
from pathlib import Path

MSU_ID = 344
MSU_FILE = "msu.png"
LOGO_URL = "https://a.espncdn.com/i/teamlogos/ncaa/500/{id}.png"


def logo_file(team_id: int) -> str:
    return MSU_FILE if int(team_id) == MSU_ID else f"{int(team_id)}.png"


def default_http(url: str) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": "msu-ticket-model/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except (urllib.error.URLError, TimeoutError):
        return 0, b""


def ensure_logos(team_ids, logos_dir, http=None) -> list[int]:
    """Download any logo not already in `logos_dir`. Returns the ids that could not be fetched;
    a missing logo is a warning for the site, never an error."""
    http = http or default_http
    logos_dir = Path(logos_dir)
    logos_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for team_id in sorted({int(t) for t in team_ids if int(t) != MSU_ID}):
        target = logos_dir / logo_file(team_id)
        if target.exists():
            continue
        status, body = http(LOGO_URL.format(id=team_id))
        if status == 200 and body:
            target.write_bytes(body)
        else:
            failed.append(team_id)
    return failed
