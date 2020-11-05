


class Delay(object):
    def __init__(self,func,*args,**kwargs):
        self._func=func
        self._args=args
        self._kwargs=kwargs

    def __call__(self):
        return self._func(*self._args,**self._kwargs)
