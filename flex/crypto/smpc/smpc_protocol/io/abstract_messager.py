"""Abstract messageer
"""

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

from abc import ABC
from abc import abstractmethod


class AbstractMessager(ABC):
    """Base class which supports send_msg and receive_msg
    """
    @abstractmethod
    def send_msg(self, dst, msg, tag):
        """Sends message from one party to another.

        As AbstractMessager implies, you should never instantiate thsi class by
        itself. Instead, you should extend AbstractMessager in new class which
        instantiates send_msg and receive_msg.

        Args:
            dst: The destination parry.
            msg: The item to be sent.
            tag: The tag to discriminate from other messages.
        """
        return

    @abstractmethod
    def receive_msg(self, tag, src):
        """Sends message from one party to another.

        As AbstractMessager implies, you should never instantiate thsi class by
        itself. Instead, you should extend AbstractMessager in new class which
        instantiates send_msg and receive_msg.

        Args:
            tag: The tag to discriminate from other messages.
            src: The party/arbiter who sent the message.
        """
        return
