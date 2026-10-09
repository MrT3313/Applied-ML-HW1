from pathlib import Path

PROJECT_DIR = Path('/Users/mrt/Documents/MrT/IUBloomington/Fall 2026 - Applied Machine Learning/code/hw/Applied-ML-HW1')
DATA_PATH = PROJECT_DIR / 'Q1' / 'DATA'
TEST_PATH = DATA_PATH / 'happiness_data_test.csv'
TRAIN_PATH = DATA_PATH / 'happiness_data_train.csv'
MODEL_OUTPUT_DIR = PROJECT_DIR / 'Q1' / 'MODELS'

COLUMNS = {
	'LL': { "col": 'Life Ladder', "short_name": "Ladder" },
	'LGDP': { "col": 'Log GDP per capita', "short_name": "GDP" },
	'SS': { "col": 'Social support', "short_name": "Social" },
	'HLEAB': {"col": 'Healthy life expectancy at birth', "short_name": "Life exp" },
	'FTMLC': {"col": 'Freedom to make life choices', "short_name": "Freedom" },
	'G': { "col": 'Generosity', "short_name": "Generosity" },
	"POC": {"col": 'Perceptions of corruption', "short_name": "Corruption"},
	'PA': {"col": 'Positive affect', "short_name": "Pos affect"},
	'NA': {"col": 'Negative affect', "short_name": "Neg affect"},
}
TARGET = COLUMNS['LL']['col']
SHORT_NAMES = {c['col']: c['short_name'] for c in COLUMNS.values()}