#!/usr/bin/python3
#
#  Copyright 2020 The iBond Authors @AI Institute, Tongdun Technology.
#  All Rights Reserved.
#                                                                                              
#  Project name: iBond                                                                         
#                                                                                              
#  File name: alignment.py                                                                          
#                                                                                              
#  Create date: 2020/11/26                                                                               
#
from logging import getLogger
from typing import List, Tuple, Union

from ..utils import ClassMethodAutoLog


class MockAligner(object):
    def __init__(self, meta_param, session):
        self.logger = getLogger(self.__class__.__name__)
        self.logger.info(f'in MockAligner.__init__')
        self.wafer = session

    @ClassMethodAutoLog()
    def align(self, id_cols: List[List[str]], place_holder: str='^_^') -> Union[None, List[List[str]]]:
        # NOTE no check special char in ids
        # run align
        return None

