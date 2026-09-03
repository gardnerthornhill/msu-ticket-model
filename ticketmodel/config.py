"""Project-wide constants and paths."""
from dataclasses import dataclass
from pathlib import Path

TEAM = "Mississippi State"
VENUE = "Davis Wade Stadium"
CAPACITY = 60417  # observed sellout figure; official listed capacity is 60,311
TZ = "America/Chicago"
P4_CONFERENCES = {"SEC", "Big Ten", "Big 12", "ACC"}

# ticketdata opponent name (exact, or prefix followed by a space) -> CFBD team name
ALIASES = {
    "UMass": "Massachusetts",
    "Southeastern Louisiana": "SE Louisiana",
    "USM": "Southern Miss",
    "Southern Mississippi": "Southern Miss",
}

CANDIDATE_FEATURES = ["opp_ranked", "conf_game", "opp_elo", "opp_sp", "opp_p4", "week"]
PRICE_FEATURES = ["log_getin", "rel_log_price"]
MAX_SUBSET_SIZE = 3
SELECTION_TOLERANCE = 0.05  # prefer fewer features when LOO-RMSE is within this fraction of the best
MIN_TRAINING_ROWS = 8
MIN_PRICED_PER_SEASON = 3  # Tier 2 predictions need this many priced games in the game's season
REFRESH_WINDOW_DAYS = 14
INTERVAL = 0.80  # prediction interval coverage


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cfbd_raw"

    @property
    def tickets(self) -> Path:
        return self.data_dir / "tickets.csv"

    @property
    def features(self) -> Path:
        return self.data_dir / "features.csv"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def tier1(self) -> Path:
        return self.models_dir / "tier1.json"

    @property
    def tier2(self) -> Path:
        return self.models_dir / "tier2.json"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def report(self) -> Path:
        return self.reports_dir / "model_report.md"

    @property
    def predictions(self) -> Path:
        return self.reports_dir / "predictions.csv"


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATHS = Paths(ROOT)
