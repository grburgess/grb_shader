from typing import List

class BaseProfile(object):

    def __init__(self, ep_profile=None, **params) -> None:

        self._qauntites: List = None

        if ep_profile is None:
            self._construct(**params)
        else:
            self._construct(ep_profile,**params)

    def _construct(self):

        pass