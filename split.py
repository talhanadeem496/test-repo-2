import subprocess
import sys


# Function to install required packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
try:
    import polars as pl
    from sklearn.model_selection import train_test_split
    import numpy as np
except:
    # Install necessary packages
    install('polars')
    install('scikit-learn')
    install('numpy')

    import polars as pl
    from sklearn.model_selection import train_test_split
    import numpy as np

class DataFrameSplitter:
    def __init__(self, start_index, end_index, split_percentage, shuffle, random_seed, max_ram_usage):
        """
        Initializes the DataFrameSplitter with the given parameters.

        Parameters:
        start_index (int): The starting index for the split.
        end_index (int): The ending index for the split.
        split_percentage (float): The percentage of the data to be used for the test set (between 0 and 1).
        shuffle (bool): Whether to shuffle the data before splitting.
        random_seed (int): The seed for random number generation to ensure reproducibility.
        max_ram_usage (int): The maximum amount of RAM (in MB) to use for batching.
        """
        self.start_index = start_index
        self.end_index = end_index
        self.split_percentage = split_percentage
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.max_ram_usage = max_ram_usage

    def split_data(self, input_table):
        """
        Splits the input table into training and test tables based on the provided parameters.

        Parameters:
        input_table (pl.DataFrame): The input table to be split.

        Returns:
        training_table (pl.DataFrame): The training set.
        test_table (pl.DataFrame): The test set.
        """
        # Validate split_percentage
        if self.split_percentage < 0 or self.split_percentage > 1:
            raise ValueError("split_percentage should be between 0 and 1")
        
        # Slice the input_table based on start_index and end_index
        sliced_table = input_table.slice(self.start_index, self.end_index - self.start_index)
        
        # Estimate the batch size based on max_ram_usage
        # Assuming each row is approximately the same size
        row_size_bytes = sliced_table.estimated_size() / len(sliced_table)
        batch_size = int((self.max_ram_usage * 1024 * 1024) / row_size_bytes)

        # Initialize empty DataFrames for training and test sets
        training_table = pl.DataFrame()
        test_table = pl.DataFrame()

        # Process the data in batches
        np.random.seed(self.random_seed)

        for i in range(0, len(sliced_table), batch_size):
            batch = sliced_table.slice(i, min(batch_size, len(sliced_table) - i))

            if self.shuffle:
                batch = batch.sample(fraction=1, with_replacement=False, shuffle=True, seed=self.random_seed)

            # Calculate split index
            split_idx = int(len(batch) * (1 - self.split_percentage))

            # Split the batch
            train_batch = batch[:split_idx]
            test_batch = batch[split_idx:]

            # Concatenate results
            training_table = pl.concat([training_table, train_batch])
            test_table = pl.concat([test_table, test_batch])

        return training_table, test_table

# Example usage
if __name__ == "__main__":
    # Sample DataFrame for demonstration
    input_data = {
        'feature1': range(1000),
        'feature2': range(1000, 2000),
        'label': range(2000, 3000)
    }

    input_table = pl.DataFrame(input_data)

    # Define parameters
    start_index = 0
    end_index = 1000
    split_percentage = 0.2
    shuffle = True
    random_seed = 42
    max_ram_usage = 10  # in MB

    # Initialize the DataFrameSplitter
    splitter = DataFrameSplitter(start_index, end_index, split_percentage, shuffle, random_seed, max_ram_usage)

    # Split the data
    training_table, test_table = splitter.split_data(input_table)

    # Display the results
    print("Training Table:")
    print(training_table)
    print("\nTest Table:")
    print(test_table)
