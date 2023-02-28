#!/usr/bin/python3
#
#  _____                     _               _______                 _   _____        __  __     _
# |_   _|                   | |             (_) ___ \               | | /  __ \      / _|/ _|   (_)
#   | | ___  _ __   __ _  __| |_   _ _ __    _| |_/ / ___  _ __   __| | | /  \/ __ _| |_| |_ ___ _ _ __   ___
#   | |/ _ \| '_ \ / _` |/ _` | | | | '_ \  | | ___ \/ _ \| '_ \ / _` | | |    / _` |  _|  _/ _ \ | '_ \ / _ \
#   | | (_) | | | | (_| | (_| | |_| | | | | | | |_/ / (_) | | | | (_| | | \__/\ (_| | | | ||  __/ | | | |  __/
#   \_/\___/|_| |_|\__, |\__,_|\__,_|_| |_| |_\____/ \___/|_| |_|\__,_|  \____/\__,_|_| |_| \___|_|_| |_|\___|
#                   __/ |
#                  |___/
#
#  Copyright 2020 The iBond Authors @AI Institute, Tongdun Technology.
#  All Rights Reserved.
#
#  Project name: iBond                                                                      
#                                                                                              
#  File name: data.py
#                                                                                              
#  Create date: 2020/01/13                                                                              

from typing import Optional

from caffeine.utils import IBondDataFrame


def parse_ibonddf(df: Optional[IBondDataFrame]):
    if df is None:
        return {}
    nonfeat_cols = df.data_desc['id_desc'] + df.data_desc['y_desc']
    feat_cols = [n for n in df.columns if n not in nonfeat_cols]
    return {
        'nonfeat_cols': nonfeat_cols,
        'feat_cols': feat_cols,
        'y_cols': df.data_desc['y_desc'],
        'id_cols': df.data_desc['id_desc']
    }
