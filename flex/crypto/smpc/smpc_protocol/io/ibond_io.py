"""Communication using ionic_bond
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

import logging

# from flex.utils.ionic import commu
# from flex.utils.ionic import VariableChannel
from flex.cores.commu_model import CommuModel
from flex.crypto.smpc.smpc_protocol.\
    config.configuration import Configuration
from flex.crypto.smpc.smpc_protocol.\
    io.abstract_messager import AbstractMessager


class IBondIOTH(AbstractMessager, Configuration):
    """Communication Torch backend
    """
    def __init__(self, jobid, conf_file, ibond_conf):
        super().__init__(jobid, conf_file, ibond_conf)
        self.jobid = jobid
        self.ibond_conf['session']['job_id'] = jobid
        self.commu = CommuModel(self.ibond_conf)
        # commu.init(self.ibond_conf)

    def send_msg(self, dst, msg, tag):
        """Send message
        """
        remote_id = self.wid2ibond[dst]
        # var_chan = VariableChannel(name=tag,
        #                            remote_id=remote_id,
        #                            auto_offset=False)
        var_chan = self.commu.make_raw_channel(channel_name=tag,
                                               endpoint1=remote_id,
                                               endpoint2=self.wid2ibond[self.world_id])
        var_chan.send(msg, tag=tag)

        logging.info('sent_msg %s to %s', tag, dst)

    def receive_msg(self, tag, src):
        """Receive message
        """
        logging.info('waiting %s', tag)
        remote_id = self.wid2ibond[src]
        # var_chan = VariableChannel(name=tag,
        #                            remote_id=remote_id,
        #                            auto_offset=False)
        var_chan = self.commu.make_raw_channel(channel_name=tag,
                                               endpoint1=remote_id,
                                               endpoint2=self.wid2ibond[self.world_id])
        msg = var_chan.recv(tag=tag)
        logging.info('received %s from %s', tag, src)
        return msg
