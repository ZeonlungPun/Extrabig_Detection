import numpy as np
import pandas as pd
from ExtractFourLinesNum import GetFourLinesNum

#對初始的輸出結果添加對應的行和列數
def Extract_column_row_in_initial_count(csv_name,start_range=1):
    """
    :param csv_name:  initial count file
    :return:  count  with  row and column
    """
    data = pd.read_csv(csv_name)
    y1, y2 = np.sort(data['y1'].unique()), np.sort(data['y2'].unique())

    row = len(y1)+1+(start_range-1)
    count_with_column_row = pd.DataFrame()
    for y1_, y2_ in zip(y1, y2):
        row -= 1
        # get all the data in one row
        subdata = data[(data['y1'] == y1_) & (data['y2'] == y2_)]
        subdata = subdata[subdata['x1'] != subdata['x2']]
        subdata = subdata.sort_values(by='x1')
        subdata['row'] = row
        subdata['column'] = subdata['x1'].argsort() + 1
        count_with_column_row = pd.concat([count_with_column_row, subdata], axis=0)

    return count_with_column_row


# 根據行數和列數提取出對應的植株數量
def GetStandardOutput(csv_name, location, location_file):
    """
    get the final standard output
    :param csv_name: initial count result
    :param location: target location
    :param location_file:  a file containing four region/ two region information
    :return:
    """
    two_line_region_output, four_line_region_output = GetFourLinesNum(location_file, location)
    count_with_column_row = Extract_column_row_in_initial_count(csv_name)
    two_line_count = pd.DataFrame()
    # get the result of two line region
    for index, data_line in two_line_region_output.iterrows():
        range_, column_begin, column_end = data_line[0], data_line[2], data_line[2] + 1
        count_result1 = count_with_column_row[(count_with_column_row['row'] == range_) & (
                    count_with_column_row['column'] == column_begin)].loc[:, 'count']
        count_result2 = count_with_column_row[(count_with_column_row['row'] == range_) & (
                count_with_column_row['column'] == column_end)].loc[:, 'count']
        warning1 = count_with_column_row[(count_with_column_row['row'] == range_) & (
                    count_with_column_row['column'] == column_begin)].loc[:, 'flag']
        warning2 = count_with_column_row[(count_with_column_row['row'] == range_) & (
                count_with_column_row['column'] == column_end)].loc[:, 'flag']

        try:
            warning = int(warning1.iloc[0]) + int(warning2.iloc[0])
            count_result = int(count_result1.iloc[0]) + int(count_result2.iloc[0])
            warning = int(warning1.iloc[0]) + int(warning2.iloc[0])
            line_for_output = pd.DataFrame([np.array([range_, data_line[1],column_begin ,count_result, warning])],
                                           columns=['Range', 'pass', 'column_begin','final_count', 'warning'])
            two_line_count = pd.concat([two_line_count, line_for_output], axis=0)
        except:
            print('bug in {}:{}'.format(index, data_line))
    # get the result of four line region
    four_line_count = pd.DataFrame()
    for index, data_line in four_line_region_output.iterrows():
        range_, column_begin, column_end = data_line[0], data_line[2] + 1, data_line[2] + 2
        count_result1 = count_with_column_row[(count_with_column_row['row'] == range_) & (
                count_with_column_row['column'] == column_begin)].loc[:, 'count']
        count_result2 = count_with_column_row[(count_with_column_row['row'] == range_) & (
                count_with_column_row['column'] == column_end)].loc[:, 'count']
        warning1 = count_with_column_row[(count_with_column_row['row'] == range_) & (
                    count_with_column_row['column'] == column_begin)].loc[:, 'flag']
        warning2 = count_with_column_row[(count_with_column_row['row'] == range_) & (
                count_with_column_row['column'] == column_end)].loc[:, 'flag']

        try:
            count_result = int(count_result1.iloc[0]) + int(count_result2.iloc[0])
            warning = int(warning1.iloc[0]) + int(warning2.iloc[0])
            line_for_output = pd.DataFrame([np.array([range_, data_line[1], column_begin ,count_result, warning])],
                                           columns=['Range', 'pass', 'column_begin','final_count', 'warning'])
            four_line_count = pd.concat([four_line_count, line_for_output], axis=0)
        except:
            print('bug in {}:{}'.format(index, data_line))

    final_output = pd.concat([four_line_count, two_line_count], axis=0)
    final_output = final_output.sort_values(by=['Range', 'pass'])
    return final_output,two_line_count,four_line_count



if __name__ == '__main__':
    csv_name='/home/kingargroo/streamline/LINHAI1/countresult1.csv'
    data_file = '/home/kingargroo/streamline/data.csv'
    location = '辽宁凌海'
    final_output,two_line_count,four_line_count=GetStandardOutput(csv_name, location, data_file)
    print(two_line_count)




