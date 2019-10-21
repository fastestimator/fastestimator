# Copyright 2019 The FastEstimator Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Trace to send Slack messages."""
import os
import types

import nest_asyncio
import slack
from slack.errors import SlackApiError

from fastestimator.trace import Trace


class SlackNotification(Trace):
    """Send message to Slack channel when training begins and ends.

    In order to send messages to Slack, user needs to generate a Slack token for authentication. Once a token is
    generated, assign it to the class initializer argument `token`, or to the environment variable `SLACK_TOKEN`.

    For Slack token generation, see:
    https://api.slack.com/custom-integrations/legacy-tokens

    Args:
        channel (str): A string. Can be either channel name or user id.
        end_msg (Union[str, function]): The message to send to the Slack channel when training starts, can be either a
            string or a function. If this is a function, it can take the state dict as input.
        begin_msg (str, optional): The message to send to the Slack channel when training starts. Defaults to None.
        token (str, optional): This token can be generated from Slack API. Defaults to None. When the value is None,
            this argument will read from the environment variable `SLACK_TOKEN`.

    Raises:
        TypeError: If `begin_msg` or `end_msg` is not (str, function).

    """
    def __init__(self, channel, end_msg, begin_msg=None, token=None, verbose=0):
        super().__init__()

        if begin_msg:
            self._check_str_or_function(begin_msg)
        self._check_str_or_function(end_msg)

        if token is None:
            token = os.environ['SLACK_TOKEN']

        nest_asyncio.apply()
        self.client = slack.WebClient(token=token)
        self.channel = channel

        self.begin_msg = begin_msg
        self.end_msg = end_msg
        self.verbose = verbose

    @staticmethod
    def _check_str_or_function(obj):
        if not isinstance(obj, (str, types.FunctionType)):
            raise TypeError("begin_msg and end_msg must be string or function.")

    def _send_message(self, msg):
        try:
            self.client.chat_postMessage(channel=self.channel, text=msg)
        except SlackApiError as err:
            if self.verbose > 0:
                print(err)
            print("Slack API error. Continue training without sending Slack notification.")

    def on_begin(self, state):
        if isinstance(self.begin_msg, str):
            self._send_message(self.begin_msg)
        else:
            self._send_message(self.begin_msg(state))

    def on_end(self, state):
        if isinstance(self.end_msg, str):
            self._send_message(self.end_msg)
        else:
            self._send_message(self.end_msg(state))
