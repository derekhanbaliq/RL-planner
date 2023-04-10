"""
    MEAM 517 Final Project - Pure Pursuit - main application
    Author: Derek Zhou & Tancy Zhao
    References: https://xlsxwriter.readthedocs.io/
                https://f1tenth-gym.readthedocs.io/en/latest/index.html
"""

import xlsxwriter


def xlsx_log_action(map_name, actions):
    workbook = xlsxwriter.Workbook('../log/pure_pursuit/' + map_name + '_action.xlsx')
    worksheet = workbook.add_worksheet()

    worksheet.write_row('A1:A3', ['time', 'speed', 'steer'])

    col = 0
    for row, action in enumerate(actions):  # iterating through content list
        worksheet.write_row(row + 1, col, action)

    workbook.close()


def xlsx_log_observation(map_name, observations):
    workbook = xlsxwriter.Workbook('../log/pure_pursuit/' + map_name + '_observation.xlsx')
    worksheet = workbook.add_worksheet()

    worksheet.write_row('A1:A5', ['time', 'x', 'y', 'theta', 'v_x'])

    col = 0
    for row, obs in enumerate(observations):  # iterating through content list
        worksheet.write_row(row + 1, col, obs)

    workbook.close()
