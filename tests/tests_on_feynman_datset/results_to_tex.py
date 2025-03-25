import pandas as pd

if __name__ == '__main__':
    ginn_results = pd.read_csv("ginn_lp_feynmann_results.csv")
    parfam_results = pd.read_csv("parfam_feynmann_results.csv")

    merged_results = pd.concat([ginn_results, parfam_results.iloc[:,2:]], axis=1, join="inner")
    new_column_names = ['Filename', 'Target Formula', 'G: Recovered Formula', 'G: Last Error', 'G: Run Time', 'P: Recovered Formula', 'P: Last Error', 'P: Run Time']
    merged_results.columns = new_column_names
    with open('merged_results.tex', 'w', newline='') as f:
        f.write(merged_results.to_latex(index=False))