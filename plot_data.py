import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from collections import namedtuple
Location = namedtuple('Location', ['x', 'y'])

def run():
    asd = Location(1,2)
    print(asd)

if __name__ == "__main__":
    run()