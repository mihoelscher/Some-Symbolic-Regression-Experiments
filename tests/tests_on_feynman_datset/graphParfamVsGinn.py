import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    # Load data from CSV
    df = pd.read_csv("mergedResults.csv")

    # Convert boolean columns
    df['G_Rec'] = df['G_Rec'].astype(bool)
    df['P_Rec'] = df['P_Rec'].astype(bool)

    # Calculate average run times
    total_avg_g = df['G_Run_Time'].mean()
    total_avg_p = df['P_Run_Time'].mean()

    avg_g_true = df[df['G_Rec'] == True]['G_Run_Time'].mean()
    avg_p_true = df[df['P_Rec'] == True]['P_Run_Time'].mean()

    avg_g_false = df[df['G_Rec'] == False]['G_Run_Time'].mean()
    avg_p_false = df[df['P_Rec'] == False]['P_Run_Time'].mean()

    # Calculate recovery rates
    g_recovery_rate = df['G_Rec'].mean()
    p_recovery_rate = df['P_Rec'].mean()

    print("Average Run Times:")
    print(f"G Total: {total_avg_g:.2f}, P Total: {total_avg_p:.2f}")
    print(f"G Recovered True: {avg_g_true:.2f}, P Recovered True: {avg_p_true:.2f}")
    print(f"G Recovered False: {avg_g_false:.2f}, P Recovered False: {avg_p_false:.2f}")

    print("Recovery Rates:")
    print(f"G Recovery Rate: {g_recovery_rate:.2%}")
    print(f"P Recovery Rate: {p_recovery_rate:.2%}")

    # Set up first plot (Run Time Comparison)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Formula', y='G_Run_Time', color='blue', label='G Run Time')
    sns.barplot(data=df, x='Formula', y='P_Run_Time', color='orange', label='P Run Time', alpha=0.7)
    plt.xticks(rotation=90)
    plt.ylabel('Run Time (seconds)')
    plt.xlabel('Formula')
    plt.title('Comparison of G and P Run Time')
    plt.legend()
    plt.show()

    # Set up second plot (Recovery Rate Comparison)
    plt.figure(figsize=(6, 6))
    sns.barplot(x=["G Recovery", "P Recovery"], y=[g_recovery_rate, p_recovery_rate], palette=["blue", "orange"])
    plt.ylabel("Recovery Rate")
    plt.title("Recovery Rate Comparison")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
