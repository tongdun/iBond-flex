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

import secrets
from typing import List

from gmpy2 import powmod

from flex.cores.commu_model import make_raw_channel
from flex.utils import ClassMethodAutoLog


class DiffieHellman(object):
    """
    Implement Diffie-Hellman key exchange protocol,
    Parties will get the same secret key when the protocol is finished.
    Reference from RFC 7919.
    """
    @ClassMethodAutoLog()
    def __init__(self, key_length: int = 2048) -> None:
        """
        Set expected key length.

        Args:
            key_length: int, key length, must in [2048, 3072, 4096, 6144, 8192]
        """
        self.key_lengths = [2048, 3072, 4096, 6144, 8192]
        primes = [
            'FFFFFFFF FFFFFFFF ADF85458 A2BB4A9A AFDC5620 273D3CF1'
            'D8B9C583 CE2D3695 A9E13641 146433FB CC939DCE 249B3EF9'
            '7D2FE363 630C75D8 F681B202 AEC4617A D3DF1ED5 D5FD6561'
            '2433F51F 5F066ED0 85636555 3DED1AF3 B557135E 7F57C935'
            '984F0C70 E0E68B77 E2A689DA F3EFE872 1DF158A1 36ADE735'
            '30ACCA4F 483A797A BC0AB182 B324FB61 D108A94B B2C8E3FB'
            'B96ADAB7 60D7F468 1D4F42A3 DE394DF4 AE56EDE7 6372BB19'
            '0B07A7C8 EE0A6D70 9E02FCE1 CDF7E2EC C03404CD 28342F61'
            '9172FE9C E98583FF 8E4F1232 EEF28183 C3FE3B1B 4C6FAD73'
            '3BB5FCBC 2EC22005 C58EF183 7D1683B2 C6F34A26 C1B2EFFA'
            '886B4238 61285C97 FFFFFFFF FFFFFFFF',

            'FFFFFFFF FFFFFFFF ADF85458 A2BB4A9A AFDC5620 273D3CF1'
            'D8B9C583 CE2D3695 A9E13641 146433FB CC939DCE 249B3EF9'
            '7D2FE363 630C75D8 F681B202 AEC4617A D3DF1ED5 D5FD6561'
            '2433F51F 5F066ED0 85636555 3DED1AF3 B557135E 7F57C935'
            '984F0C70 E0E68B77 E2A689DA F3EFE872 1DF158A1 36ADE735'
            '30ACCA4F 483A797A BC0AB182 B324FB61 D108A94B B2C8E3FB'
            'B96ADAB7 60D7F468 1D4F42A3 DE394DF4 AE56EDE7 6372BB19'
            '0B07A7C8 EE0A6D70 9E02FCE1 CDF7E2EC C03404CD 28342F61'
            '9172FE9C E98583FF 8E4F1232 EEF28183 C3FE3B1B 4C6FAD73'
            '3BB5FCBC 2EC22005 C58EF183 7D1683B2 C6F34A26 C1B2EFFA'
            '886B4238 611FCFDC DE355B3B 6519035B BC34F4DE F99C0238'
            '61B46FC9 D6E6C907 7AD91D26 91F7F7EE 598CB0FA C186D91C'
            'AEFE1309 85139270 B4130C93 BC437944 F4FD4452 E2D74DD3'
            '64F2E21E 71F54BFF 5CAE82AB 9C9DF69E E86D2BC5 22363A0D'
            'ABC52197 9B0DEADA 1DBF9A42 D5C4484E 0ABCD06B FA53DDEF'
            '3C1B20EE 3FD59D7C 25E41D2B 66C62E37 FFFFFFFF FFFFFFFF',

            'FFFFFFFF FFFFFFFF ADF85458 A2BB4A9A AFDC5620 273D3CF1'
            'D8B9C583 CE2D3695 A9E13641 146433FB CC939DCE 249B3EF9'
            '7D2FE363 630C75D8 F681B202 AEC4617A D3DF1ED5 D5FD6561'
            '2433F51F 5F066ED0 85636555 3DED1AF3 B557135E 7F57C935'
            '984F0C70 E0E68B77 E2A689DA F3EFE872 1DF158A1 36ADE735'
            '30ACCA4F 483A797A BC0AB182 B324FB61 D108A94B B2C8E3FB'
            'B96ADAB7 60D7F468 1D4F42A3 DE394DF4 AE56EDE7 6372BB19'
            '0B07A7C8 EE0A6D70 9E02FCE1 CDF7E2EC C03404CD 28342F61'
            '9172FE9C E98583FF 8E4F1232 EEF28183 C3FE3B1B 4C6FAD73'
            '3BB5FCBC 2EC22005 C58EF183 7D1683B2 C6F34A26 C1B2EFFA'
            '886B4238 611FCFDC DE355B3B 6519035B BC34F4DE F99C0238'
            '61B46FC9 D6E6C907 7AD91D26 91F7F7EE 598CB0FA C186D91C'
            'AEFE1309 85139270 B4130C93 BC437944 F4FD4452 E2D74DD3'
            '64F2E21E 71F54BFF 5CAE82AB 9C9DF69E E86D2BC5 22363A0D'
            'ABC52197 9B0DEADA 1DBF9A42 D5C4484E 0ABCD06B FA53DDEF'
            '3C1B20EE 3FD59D7C 25E41D2B 669E1EF1 6E6F52C3 164DF4FB'
            '7930E9E4 E58857B6 AC7D5F42 D69F6D18 7763CF1D 55034004'
            '87F55BA5 7E31CC7A 7135C886 EFB4318A ED6A1E01 2D9E6832'
            'A907600A 918130C4 6DC778F9 71AD0038 092999A3 33CB8B7A'
            '1A1DB93D 7140003C 2A4ECEA9 F98D0ACC 0A8291CD CEC97DCF'
            '8EC9B55A 7F88A46B 4DB5A851 F44182E1 C68A007E 5E655F6A'
            'FFFFFFFF FFFFFFFF',

            'FFFFFFFF FFFFFFFF ADF85458 A2BB4A9A AFDC5620 273D3CF1'
            'D8B9C583 CE2D3695 A9E13641 146433FB CC939DCE 249B3EF9'
            '7D2FE363 630C75D8 F681B202 AEC4617A D3DF1ED5 D5FD6561'
            '2433F51F 5F066ED0 85636555 3DED1AF3 B557135E 7F57C935'
            '984F0C70 E0E68B77 E2A689DA F3EFE872 1DF158A1 36ADE735'
            '30ACCA4F 483A797A BC0AB182 B324FB61 D108A94B B2C8E3FB'
            'B96ADAB7 60D7F468 1D4F42A3 DE394DF4 AE56EDE7 6372BB19'
            '0B07A7C8 EE0A6D70 9E02FCE1 CDF7E2EC C03404CD 28342F61'
            '9172FE9C E98583FF 8E4F1232 EEF28183 C3FE3B1B 4C6FAD73'
            '3BB5FCBC 2EC22005 C58EF183 7D1683B2 C6F34A26 C1B2EFFA'
            '886B4238 611FCFDC DE355B3B 6519035B BC34F4DE F99C0238'
            '61B46FC9 D6E6C907 7AD91D26 91F7F7EE 598CB0FA C186D91C'
            'AEFE1309 85139270 B4130C93 BC437944 F4FD4452 E2D74DD3'
            '64F2E21E 71F54BFF 5CAE82AB 9C9DF69E E86D2BC5 22363A0D'
            'ABC52197 9B0DEADA 1DBF9A42 D5C4484E 0ABCD06B FA53DDEF'
            '3C1B20EE 3FD59D7C 25E41D2B 669E1EF1 6E6F52C3 164DF4FB'
            '7930E9E4 E58857B6 AC7D5F42 D69F6D18 7763CF1D 55034004'
            '87F55BA5 7E31CC7A 7135C886 EFB4318A ED6A1E01 2D9E6832'
            'A907600A 918130C4 6DC778F9 71AD0038 092999A3 33CB8B7A'
            '1A1DB93D 7140003C 2A4ECEA9 F98D0ACC 0A8291CD CEC97DCF'
            '8EC9B55A 7F88A46B 4DB5A851 F44182E1 C68A007E 5E0DD902'
            '0BFD64B6 45036C7A 4E677D2C 38532A3A 23BA4442 CAF53EA6'
            '3BB45432 9B7624C8 917BDD64 B1C0FD4C B38E8C33 4C701C3A'
            'CDAD0657 FCCFEC71 9B1F5C3E 4E46041F 388147FB 4CFDB477'
            'A52471F7 A9A96910 B855322E DB6340D8 A00EF092 350511E3'
            '0ABEC1FF F9E3A26E 7FB29F8C 183023C3 587E38DA 0077D9B4'
            '763E4E4B 94B2BBC1 94C6651E 77CAF992 EEAAC023 2A281BF6'
            'B3A739C1 22611682 0AE8DB58 47A67CBE F9C9091B 462D538C'
            'D72B0374 6AE77F5E 62292C31 1562A846 505DC82D B854338A'
            'E49F5235 C95B9117 8CCF2DD5 CACEF403 EC9D1810 C6272B04'
            '5B3B71F9 DC6B80D6 3FDD4A8E 9ADB1E69 62A69526 D43161C1'
            'A41D570D 7938DAD4 A40E329C D0E40E65 FFFFFFFF FFFFFFFF',

            'FFFFFFFF FFFFFFFF ADF85458 A2BB4A9A AFDC5620 273D3CF1'
            'D8B9C583 CE2D3695 A9E13641 146433FB CC939DCE 249B3EF9'
            '7D2FE363 630C75D8 F681B202 AEC4617A D3DF1ED5 D5FD6561'
            '2433F51F 5F066ED0 85636555 3DED1AF3 B557135E 7F57C935'
            '984F0C70 E0E68B77 E2A689DA F3EFE872 1DF158A1 36ADE735'
            '30ACCA4F 483A797A BC0AB182 B324FB61 D108A94B B2C8E3FB'
            'B96ADAB7 60D7F468 1D4F42A3 DE394DF4 AE56EDE7 6372BB19'
            '0B07A7C8 EE0A6D70 9E02FCE1 CDF7E2EC C03404CD 28342F61'
            '9172FE9C E98583FF 8E4F1232 EEF28183 C3FE3B1B 4C6FAD73'
            '3BB5FCBC 2EC22005 C58EF183 7D1683B2 C6F34A26 C1B2EFFA'
            '886B4238 611FCFDC DE355B3B 6519035B BC34F4DE F99C0238'
            '61B46FC9 D6E6C907 7AD91D26 91F7F7EE 598CB0FA C186D91C'
            'AEFE1309 85139270 B4130C93 BC437944 F4FD4452 E2D74DD3'
            '64F2E21E 71F54BFF 5CAE82AB 9C9DF69E E86D2BC5 22363A0D'
            'ABC52197 9B0DEADA 1DBF9A42 D5C4484E 0ABCD06B FA53DDEF'
            '3C1B20EE 3FD59D7C 25E41D2B 669E1EF1 6E6F52C3 164DF4FB'
            '7930E9E4 E58857B6 AC7D5F42 D69F6D18 7763CF1D 55034004'
            '87F55BA5 7E31CC7A 7135C886 EFB4318A ED6A1E01 2D9E6832'
            'A907600A 918130C4 6DC778F9 71AD0038 092999A3 33CB8B7A'
            '1A1DB93D 7140003C 2A4ECEA9 F98D0ACC 0A8291CD CEC97DCF'
            '8EC9B55A 7F88A46B 4DB5A851 F44182E1 C68A007E 5E0DD902'
            '0BFD64B6 45036C7A 4E677D2C 38532A3A 23BA4442 CAF53EA6'
            '3BB45432 9B7624C8 917BDD64 B1C0FD4C B38E8C33 4C701C3A'
            'CDAD0657 FCCFEC71 9B1F5C3E 4E46041F 388147FB 4CFDB477'
            'A52471F7 A9A96910 B855322E DB6340D8 A00EF092 350511E3'
            '0ABEC1FF F9E3A26E 7FB29F8C 183023C3 587E38DA 0077D9B4'
            '763E4E4B 94B2BBC1 94C6651E 77CAF992 EEAAC023 2A281BF6'
            'B3A739C1 22611682 0AE8DB58 47A67CBE F9C9091B 462D538C'
            'D72B0374 6AE77F5E 62292C31 1562A846 505DC82D B854338A'
            'E49F5235 C95B9117 8CCF2DD5 CACEF403 EC9D1810 C6272B04'
            '5B3B71F9 DC6B80D6 3FDD4A8E 9ADB1E69 62A69526 D43161C1'
            'A41D570D 7938DAD4 A40E329C CFF46AAA 36AD004C F600C838'
            '1E425A31 D951AE64 FDB23FCE C9509D43 687FEB69 EDD1CC5E'
            '0B8CC3BD F64B10EF 86B63142 A3AB8829 555B2F74 7C932665'
            'CB2C0F1C C01BD702 29388839 D2AF05E4 54504AC7 8B758282'
            '2846C0BA 35C35F5C 59160CC0 46FD8251 541FC68C 9C86B022'
            'BB709987 6A460E74 51A8A931 09703FEE 1C217E6C 3826E52C'
            '51AA691E 0E423CFC 99E9E316 50C1217B 624816CD AD9A95F9'
            'D5B80194 88D9C0A0 A1FE3075 A577E231 83F81D4A 3F2FA457'
            '1EFC8CE0 BA8A4FE8 B6855DFE 72B0A66E DED2FBAB FBE58A30'
            'FAFABE1C 5D71A87E 2F741EF8 C1FE86FE A6BBFDE5 30677F0D'
            '97D11D49 F7A8443D 0822E506 A9F4614E 011E2A94 838FF88C'
            'D68C8BB7 C5C6424C FFFFFFFF FFFFFFFF'
            ]

        dic = dict(zip(self.key_lengths, primes))

        # check key length
        self._check_key_length(key_length)

        self.prime = int(dic.get(key_length).replace(' ', ''), 16)
        self.generator = 2

        self.secrets_generator = secrets.SystemRandom()

    @ClassMethodAutoLog()
    def _check_key_length(self, key_length: int):
        """
        check key length in [2048, 3072, 4096, 6144, 8192].
        Args:
            key_length: int, key length, must in [2048, 3072, 4096, 6144, 8192]
        """
        if key_length not in self.key_lengths:
            for s in self.key_lengths:
                if key_length > s:
                    print(f"Warning: expected key size is {key_length} but supported key size is {s}.")
                    key_length = s
                    break

    @ClassMethodAutoLog()
    def key_exchange(self, remote_id: List, local_id: str, job_id: str) -> int:
        """
        N-party key exchange protocol.

        Args:
            remote_id: list, remote ID which is configured in route table.
            local_id: str, local ID.
        Returns:
            int, secret key

        -----

        **Example:**

        >>>remote_id = ["zhibang-d-011040", "zhibang-d-011041", "zhibang-d-011042"]
        >>>local_id = "zhibang-d-011040"
        >>>job_id = 'hhh'
        >>>secret_key = make_agreement(remote_id=remote_id, local_id=local_id, key_length=2048, job_id=job_id)
        """
        # sort the remote_id list to init channel
        local_index = remote_id.index(local_id)
        sorted_list = remote_id[local_index:] + remote_id[:local_index]

        # inits communication
        next_chan = make_raw_channel(
            channel_name='diffie_hellman',
            endpoint1=local_id,
            endpoint2=sorted_list[1],
            job_id=job_id
        )
        prev_chan = make_raw_channel(
            channel_name='diffie_hellman',
            endpoint1=local_id,
            endpoint2=sorted_list[-1],
            job_id=job_id
        )

        # get a random int
        rand = self.secrets_generator.randint(1, self.prime - 1)

        # send g^rand mod prime to next party
        g_power_a = powmod(self.generator, rand, self.prime)
        next_chan.send(g_power_a)

        # receive the intermediate result from prev party and calculate power rand to next party
        for i in range(len(remote_id) - 2):
            g_power_prev = prev_chan.recv()
            g_power_next = powmod(g_power_prev, rand, self.prime)
            next_chan.send(g_power_next)

        # calculate secret key
        g_power_prev = prev_chan.recv()
        secret_key = int(powmod(g_power_prev, rand, self.prime))

        return secret_key
