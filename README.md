# How to run

0. Install Python
1. Place mittausdata.csv in root directory
2. Install requirements (optionally using virtual environment)

	`pip install -r requirements.txt`

3. Execute scripts in this order:
	1. `preprocess_data.py`
	2. `process_data.py`
	3. `plot_data.py`
	You'll have to manually save figures.
4. Write report.

## Lab specific settings

File `constants.py` houses config known positions
and allows user to select certain reference points.
