# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
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

"""
LangChain CallbackHandler that prints to streamlit.

This is a special API that's imported and used by LangChain itself. Any updates
to the public API (the StreamlitCallbackHandler constructor, and the entirety
of LLMThoughtLabeler) *must* remain backwards-compatible to avoid breaking
LangChain.

This means that it's acceptable to add new optional kwargs to StreamlitCallbackHandler,
but no new positional args or required kwargs should be added, and no existing
args should be removed. If we need to overhaul the API, we must ensure that a
compatible API continues to exist.

Any major change to the StreamlitCallbackHandler should be tested by importing
the API *from LangChain itself*.
"""

from __future__ import annotations

from typing import Any, Dict

from langchain.callbacks.base import (  # type: ignore[import-not-found, unused-ignore]
    BaseCallbackHandler,
)
from langchain.schema import (  # type: ignore[import-not-found, unused-ignore]
    AgentAction,
    AgentFinish,
    LLMResult,
)

from streamlit.external.langchain.streamlit_callback_handler import LLMThought, StreamlitCallbackHandler


class CustomStreamlitCallbackHandler(StreamlitCallbackHandler):

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self._require_current_thought().on_llm_end(response, **kwargs)
        self._current_thought.complete('finish')
        self._complete_current_thought()

    def on_llm_error(self, error: BaseException, *args: Any, **kwargs: Any) -> None:
        self._require_current_thought().on_llm_error(error, **kwargs)
        self._current_thought.complete('error')
        self._complete_current_thought()

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        if self._current_thought is None:
            self._current_thought = LLMThought(
                parent_container=self._parent_container,
                expanded=self._expand_new_thoughts,
                collapse_on_complete=self._collapse_completed_thoughts,
                labeler=self._thought_labeler,
            )
        self._require_current_thought().on_tool_start(serialized, input_str, **kwargs)
