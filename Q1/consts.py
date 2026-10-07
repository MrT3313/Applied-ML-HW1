from pathlib import Path

PROJECT_DIR = Path('/Users/mrt/Documents/MrT/IUBloomington/Fall 2026 - Applied Machine Learning/code/hw/Applied-ML-HW1')
DATA_PATH = PROJECT_DIR / 'Q1' / 'DATA'
TEST_PATH = DATA_PATH / 'happiness_data_test.csv'
TRAIN_PATH = DATA_PATH / 'happiness_data_train.csv'
MODEL_OUTPUT_DIR = PROJECT_DIR / 'Q1' / 'MODELS'

COLUMNS = {
	'LL': 'Life Ladder',
	'LGDP': 'Log GDP per capita',
	'SS': 'Social support',
	'HLEAB': 'Healthy life expectancy at birth',
	'FTMLC': 'Freedom to make life choices',
	'G': 'Generosity',
	"POC": 'Perceptions of corruption',
	'PA': 'Positive affect',
	'NA': 'Negative affect',
}

SHORT_NAMES = {
    "Life Ladder": "Ladder",
    "Log GDP per capita": "GDP",
    "Social support": "Social",
    "Healthy life expectancy at birth": "Life exp",
    "Freedom to make life choices": "Freedom",
    "Generosity": "Generosity",
    "Perceptions of corruption": "Corruption",
    "Positive affect": "Pos affect",
    "Negative affect": "Neg affect",
}

HIST_BINS = 30
TARGET = COLUMNS['LL']
