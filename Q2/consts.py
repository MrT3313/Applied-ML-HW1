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
# The continuous attributes, used by the categorical interaction tables.
CONTINUOUS = ['Temp', 'FeelsLike', 'Humidity', 'Wind']

CV_TEST_SIZE = 46  # D: validation days in each of the 4 time-ordered folds, so every fold trains on some 2012 days
