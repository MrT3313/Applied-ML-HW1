from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / 'Q2' / 'DATA' / 'bike_sharing_byDay.csv'
MODEL_OUTPUT_DIR = PROJECT_DIR / 'Q2' / 'MODELS'

COLUMNS = {
	"instant": { "name": "Record Index", "short_name": "ID" },
	"dteday": { "name": "Date", "short_name": "Date" },
	"season": { "name": "Season", "short_name": "Season" },
	"yr": { "name": "Year", "short_name": "Year" },
	"mnth": { "name": "Month", "short_name": "Month" },
	"holiday": { "name": "IsHoliday", "short_name": "IsHoliday" },
	"weekday": { "name": "Weekday", "short_name": "Weekday" },
	"workingday": { "name": "IsWorking Day", "short_name": "IsWorkday" },
	"weathersit": { "name": "Weather Situation", "short_name": "Weather" },
	"temp": { "name": "Temperature", "short_name": "Temp" },
	"atemp": { "name": "Feels-Like Temperature", "short_name": "FeelsLike" },
	"hum": { "name": "Humidity", "short_name": "Humidity" },
	"windspeed": { "name": "Wind Speed", "short_name": "Wind" },
	"casual": { "name": "Casual Rentals", "short_name": "Casual" },
	"registered": { "name": "Registered Rentals", "short_name": "Registered" },
	"cnt": { "name": "Total Rentals", "short_name": "Rentals" },
}

SHORT_NAMES = {col: c['short_name'] for col, c in COLUMNS.items()}
TARGET = SHORT_NAMES["cnt"]
TARGET_UNIT = 'rentals per day'  # the unit of every RMSE, for the D and E chart labels
# The continuous attributes: used by Q2B's interaction tables, Q2C's scatter matrix and R² gain tables, the shape fits
# (FITS) of Q2D and Q2E, and the columns Q2E expands (EXPAND).
CONTINUOUS = ['Temp', 'FeelsLike', 'Humidity', 'Wind']

CV_TEST_SIZE = 46  # D and E: validation days in each of the 4 time-ordered folds (reasoning in Q2D's Method)
