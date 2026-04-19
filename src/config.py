"""Track B canonical config (geo, crop, splits, soil).

Mirrors experiments/gbdt/config.py for the main pipeline so Track A and Track B
agree on physics constants, field geometry, and evaluation windows.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_INTERIM = DATA / "interim"
DATA_PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

LAT = 44.1125
LON = 10.411

TRANSPLANT_DATE = "2023-05-01"
T_BASE_C = 10.0

GDD_STAGE_EDGES = [
    (0, 350, "initial"),
    (350, 700, "development"),
    (700, 1100, "mid"),
    (1100, 1500, "late"),
]
KC_BY_STAGE = {"initial": 0.40, "development": 0.75, "mid": 1.15, "late": 0.85}

HORIZONS_10MIN = [6, 18, 36, 72, 144]
HORIZON_LABELS = ["1h", "3h", "6h", "12h", "24h"]

LOOKBACK_HOURS = 24
LOOKBACK_STEPS = 144

TRAIN_START = "2023-07-28"
TRAIN_END = "2023-08-20 23:59:59"
VAL_START = "2023-08-21"
VAL_END = "2023-08-27 23:59:59"
TEST_START = "2023-08-28"
TEST_END = "2023-09-03 23:59:59"
EMBARGO_HOURS = 24

FIELD_CAPACITY_PCT = 27.41
WILTING_POINT_PCT = 15.0
MAD_FRACTION_BY_STAGE = {"initial": 0.30, "development": 0.40, "mid": 0.40, "late": 0.50}

LINE_FLOW_LPH = 300.0
PUMP_POWER_KW = 4.0
FIELD_AREA_M2 = 132.0

ACTION_VOLUME_MM = {"OFF": 0.0, "ON_LOW": 5.0, "ON_HIGH": 10.0}

IRRIFRAME_SEASON_MM = 324.5

HUBER_DELTA = 1.0
HORIZON_LOSS_WEIGHTS = [1.0, 0.8, 0.6, 0.4, 0.3]
