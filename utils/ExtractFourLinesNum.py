import pandas as pd
import numpy as np
def GetFourLinesNum(data_file,target_location):
    data = pd.read_csv(data_file)
    # filter location
    data = data[data['Location'] == target_location]
    # group by using range
    range_groups = data.groupby('Range')
    # a list contain range==1, range==2 , range==3 etc
    four_line_region_output = pd.DataFrame()
    two_line_region_output = pd.DataFrame()

    range_groups = list(range_groups)
    for range_group in range_groups:
        range_ = range_group[0]
        range_group = range_group[1]

        range_group=range_group.sort_values(by='pass')
        range_group['next_pass'] = range_group['pass'].shift(-1)

        range_group['TrialType'] = np.where(range_group['next_pass'] - range_group['pass'] == 2, 'Filler-4', 'Filler')
        range_group['TrialType'].iloc[-1] = 'Filler' if (
                    range_group['pass'].iloc[-1] - range_group['pass'].iloc[-2] != 2) else 'Filler-4'

        four_line_region = range_group[(range_group['TrialType'] == 'P2') | (range_group['TrialType'] == 'P3') | (
                range_group['TrialType'] == 'COMP') | (range_group['TrialType'] == 'Filler-4')]
        two_line_region = range_group[(range_group['TrialType'] == 'TC1') | (range_group['TrialType'] == 'TC2') | (
                range_group['TrialType'] == 'P1') | (range_group['TrialType'] == 'Filler')]
        four_line_region, two_line_region = four_line_region.sort_values(by='pass'), two_line_region.sort_values(
            by='pass')

        # get the corresponding column
        four_line_region['column_num_begin'] = 2 * four_line_region['pass'] - 1
        two_line_region['column_num_begin'] = 2 * two_line_region['pass'] - 1
        four_line_region_ = four_line_region.loc[:, ['Range', 'pass', 'column_num_begin']]
        two_line_region_ = two_line_region.loc[:, ['Range', 'pass', 'column_num_begin']]

        two_line_region_output = pd.concat([two_line_region_output, two_line_region_], axis=0)
        four_line_region_output = pd.concat([four_line_region_output, four_line_region_], axis=0)
    return two_line_region_output,four_line_region_output


if __name__ == '__main__':
    data_file='../data.csv'
    location='辽宁凌海'

    two_line_region_output,four_line_region_output=GetFourLinesNum(data_file, location)
    print(two_line_region_output)
    print(four_line_region_output)




