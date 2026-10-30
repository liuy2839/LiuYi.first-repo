# -*- coding: utf-8 -*-
"""pyarrow / forge_msgs 最小桩，供本地单测。"""


class _Arr:
    def __init__(self, items):
        self._items = list(items)

    def __getitem__(self, i):
        return self._items[i]

    def __len__(self):
        return len(self._items)


class array:  # noqa: N801 仿 pyarrow.array
    def __init__(self, items):
        self._items = list(items)

    def __getitem__(self, i):
        return self._items[i]

    def __len__(self):
        return len(self._items)


Array = array


class Msgs:
    class JointState:
        @classmethod
        def from_np(cls, arr, names, key):
            return cls()

        def to_arrow(self):
            return "js_arrow"

    class Image:
        @classmethod
        def from_numpy(cls, frame, enc):
            return cls()

        def to_arrow(self):
            return "img_arrow"

    class PolicyCommand:
        @classmethod
        def from_inputs(cls, policy_id, command, inputs):
            return cls()

        def to_arrow(self):
            return "cmd_arrow"
