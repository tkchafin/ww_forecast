import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Format date and rename file.')
    parser.add_argument('file_path', type=str, help='Input file path')

    args = parser.parse_args()
    file_path = args.file_path

    # Read in the file
    df = pd.read_csv(file_path, dtype=str)

    # Rename the first column to 'Date'
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    # Convert dates
    try:
        df["Date"] = pd.to_datetime(df["Date"], format='%Y%m%d')
    except ValueError:
        print("Unable to parse dates. Please check your date format.")

    # Write the formatted dataframe to a new file
    file_dir = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    new_file_name = file_name.replace(" ", "_").replace(".csv", ".formatted.csv")
    new_file_path = os.path.join(file_dir, new_file_name)

    df.to_csv(new_file_path, index=False)

if __name__ == '__main__':
    main()
