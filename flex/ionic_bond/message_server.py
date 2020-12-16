"""Communication protocol with socket connection.
It is composed of the HEAD part and the body part.
The variable name is putted in HEAD. The serielized variable is in body.
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

import asyncio
import os
import logging
import sys
import tempfile

logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s;%(levelname)s;%(message)s',
                    level=logging.DEBUG)

HEAD = 4 * 1024
SERVER = ('0.0.0.0', 16001)
PREFIX = '/dev/shm'
if not os.path.exists(PREFIX):
    PREFIX = os.path.join(tempfile.gettempdir(), 'msg_server')
if not os.path.exists(PREFIX):
    os.mkdir(PREFIX)


async def handle_client(reader, writer):
    """Message handler
        Operators:
            CLOSE                 : req remote server to close
            REQ_STORE_THEN_WAIT   : req remote to receive message
                                    and waiting for next message
            REQ_STORE_THEN_CLOSE  : req remote to receive emssage
                                    and close connection
    """
    # logging.info('connected: %s', writer.get_extra_info('peername'))
    logging.info('connected')
    payload = (await reader.readexactly(HEAD)).rstrip(b'\x00').decode()
    fname, op_code = payload.split(';')
    while True:
        if op_code == 'CLOSE':
            writer.close()
            break
        fname, fsize = fname.split('#')
        fsize = int(fsize)
        logging.info('fetching %s', fname)
        with open(f'{PREFIX}/{fname}', 'wb') as message_file:
            message_file.write(await reader.readexactly(fsize))
        os.system(f'touch "{PREFIX}/{fname}.done"')
        logging.info('received %s: size %i', fname, fsize)

        # elif op == 'REQ_STORE_THEN_WAIT':
        if op_code == 'REQ_STORE_THEN_CLOSE':
            writer.close()
            break

        try:
            payload = (await reader.readexactly(HEAD)).rstrip(b'\x00').decode()
            fname, op_code = payload.split(';')
        except asyncio.IncompleteReadError:
            logging.info('close connection')
            writer.close()
            break


def start():
    """Start server
    """
    loop = asyncio.get_event_loop()
    loop.create_task(asyncio.start_server(handle_client,
                                          *SERVER,
                                          backlog=300))
    try:
        print(f'server: {SERVER}')
        loop.run_forever()
    except KeyboardInterrupt:
        print('Bye.')
        loop.stop()


if __name__ == '__main__':
    start()
