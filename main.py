# this is entrypoint for processing
from pathlib import Path

# run preprocess
import preprocess_data as PreProcess
if not Path('preprocess_data.csv').exists():
    PreProcess.run()
# run processing
import preprocess_data as ProcessData
if Path('preprocess_data.csv').exists():
    ProcessData.run()
# create plots
import plot_data as PlotData
if Path('preprocess_data.csv').exists():
    ProcessData.run()