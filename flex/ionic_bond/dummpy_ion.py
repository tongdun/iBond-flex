#
#  Copyright 2020 The FLEX Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import time
from typing import Any, Union, List
import threading

kv_storage = {}


class Ion():
    """
    flex bond main class
    """

    @staticmethod
    def send(key: str, value: Any, dst: Union[str, List[str]]) -> bool:
        """
        Send data to remote using key.

        Arg:
            key: str, a key contains all infomation about the data.
            value: Any, local variable.
            dst: str or a list of str, each is a id of fedration.

        Return:
            bool: True for success, False for failure
        """

        def _send(key, value, dst):
            kv_storage[key] = value

        print(f"Sending {key}->{dst} in {threading.get_ident()}.")
        dst = [dst] if isinstance(dst, str) else dst
        rets = [_send(key, value, _dst) for _dst in dst]
        return all(rets)


    @staticmethod
    def recv(key: str, src: Union[str, List[str]] = None) -> Any:
        """
        Recv data from remote using key.

        Arg:
            key: str, a key contains all infomation about the data.
            src: Union[str, List[str]], Source is not used actually
                But it is kept to make a consistent API.
        Return:
            Any: pickle loaded object.
        """
        print(f"Getting {key}->{src} in {threading.get_ident()}.")
        while key not in kv_storage:
            time.sleep(0.1)

        result = kv_storage[key]
        del kv_storage[key]
        return result
