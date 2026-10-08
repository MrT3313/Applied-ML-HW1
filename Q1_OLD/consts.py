TEST_PATH = '/Q1_OLD/DATA/happiness_data_test.csv'
TRAIN_PATH = '/Q1_OLD/DATA/happiness_data_train.csv'

columns = {
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

HIST_BINS = 30
TARGET = columns['LL']
