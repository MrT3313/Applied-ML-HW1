from pathlib import Path

PROJECT_DIR = Path('/Users/mrt/Documents/MrT/IUBloomington/Fall 2026 - Applied Machine Learning/code/hw/Applied-ML-HW1')
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

TEST_SIZE = 0.2

CV_SPLITS = 6
CV_TEST_SIZE = 31

HIST_BINS = 30

SEEDS = [42, 0, 1, 2, 3]
ALPHAS = [0.001, 0.01, 0.1]
LEARNING_RATES = [0.0001, 0.001, 0.01, 0.1]
