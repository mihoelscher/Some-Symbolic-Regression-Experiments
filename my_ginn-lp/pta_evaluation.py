import pandas as pd


def analyze_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Normalize column names to avoid spaces or hidden characters
    df.columns = [col.strip() for col in df.columns]

    # Count occurrences of PTA and PTA with log
    pta_count = (df['Winner'] == 'PTA').sum()
    pta_log_count = (df['Winner']=='PTA with log').sum()
    fail_count = df['Winner'].str.count('Fail').sum()
    total_count = len(df)

    # Calculate relative frequencies
    pta_ratio = pta_count / total_count * 100
    pta_log_ratio = pta_log_count / total_count * 100
    fail_ratio = fail_count / total_count * 100

    # Print results
    print(f"Total entries: {total_count}")
    print(f"PTA count: {pta_count} ({pta_ratio:.2f}%)")
    print(f"PTA with log count: {pta_log_count} ({pta_log_ratio:.2f}%)")
    print(f"Fail count: {fail_count} ({fail_ratio:.2f}%)")

if __name__ == '__main__':
    csv_file_path = 'pta_block.csv'
    analyze_csv(csv_file_path)

### Result:
# Total entries: 3000
# PTA count: 2343 (78.10%)
# PTA with log count: 75 (2.50%)
# Fail count: 582 (19.40%)
