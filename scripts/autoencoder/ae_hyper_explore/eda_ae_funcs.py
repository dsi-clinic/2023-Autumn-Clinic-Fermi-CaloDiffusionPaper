"""Function .py file for EDA Notebook"""


import re
import pandas as pd
import matplotlib.pyplot as plt


def get_lr(filename):
    """Extracts LR from filename"""
    regex = r'\d.*\d'
    match = re.search(regex, filename)
    if match:
        return match.group()
    return None

def convert_sci_to_decimal(x):
    if ('e' in str(x) or 'E' in str(x)):
        return '{:.20f}'.format(x).rstrip('0').rstrip('.')
    return x

class Loss:
    """Creates an instance of Loss class object."""
    def __init__(self, filepath):
        """Initializes Loss class."""
        # Creates a DataFrame of loss values by Epoch
        self.df = pd.read_csv(filepath, delimiter="\t")
        self.df["loss"] = self.df["loss"].apply(convert_sci_to_decimal)
        self.type = "val_loss" if filepath.startswith("val_") else "loss"
        lr = get_lr(filepath)
        self.lr = lr if lr else ValueError(
            f"No LR specified in filepath: {filepath}"
        )

    def visualize(self, scale='linear', compare=None):
        """Displays a line plot of loss by epoch."""
        plt.plot(self.df.index, self.df["loss"], label="LR=" + self.lr)
        
        if compare:
            for obj in compare:
                plt.plot(obj.df.index, obj.df["loss"], label="LR=" + obj.lr)

        plt.xlabel("Epoch")
        plt.ylabel(self.type)
        plt.yscale(scale)
        
        if not compare:
            plt.title(
                f"{self.type} of AE Training Over {self.df.shape[0]} Epochs\nLR:{self.lr}"
            )
        else:
            plt.title(
                f"{self.type} Comparison of AE Training with Different LRs"
            )

        plt.legend()

        plt.show()

    

      
