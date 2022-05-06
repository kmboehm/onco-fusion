import pandas as pd

def make_df(input_filenames):
    # load & concatenate if necessary
    if type(input_filenames) is str:
        input_filenames = [input_filenames]

    dfs = [pd.read_csv(input_filename, engine='python') for input_filename in input_filenames]
    df = pd.concat(dfs)

    # drop NaNs and duplicates (for ovarian)
    df = df.dropna(how='any')
    df = df.drop_duplicates(subset=['Patient ID'])
    
    # remove extraneous columns
    df = df.set_index('Patient ID').filter(regex='^wavelet', axis=1)
    df = df.reset_index()

    return df


if __name__ == '__main__':
    omentum_df = make_df('features/_ct_omentum_bin25.csv')
    omentum_df.to_csv('features/ct_features_omentum.csv', index=False)

    ovary_df = make_df(['features/_ct_left_ovary_bin25.csv',
                        'features/_ct_right_ovary_bin25.csv'])
    ovary_df.to_csv('features/ct_features_ovary.csv', index=False)
