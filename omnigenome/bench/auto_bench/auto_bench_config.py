







from argparse import Namespace

from transformers import PretrainedConfig

from .config_check import config_check


class AutoBenchConfig(PretrainedConfig):
    def __init__(self, args=None, **kwargs):
        
        if not args:
            args = {}
        super().__init__(**kwargs)

        if isinstance(args, Namespace):
            self.args = vars(args)
            self.args_call_count = {arg: 0 for arg in vars(args)}
        else:
            self.args = args
            self.args_call_count = {arg: 0 for arg in args}

    def __getattribute__(self, arg_name):
        
        if arg_name == "args" or arg_name == "args_call_count":
            return super().__getattribute__(arg_name)
        try:
            value = super().__getattribute__("args")[arg_name]
            args_call_count = super().__getattribute__("args_call_count")
            args_call_count[arg_name] += 1
            super().__setattr__("args_call_count", args_call_count)
            return value

        except Exception as e:
            return super().__getattribute__(arg_name)

    def __setattr__(self, arg_name, value):
        
        if arg_name == "args" or arg_name == "args_call_count":
            super().__setattr__(arg_name, value)
            return
        try:
            args = super().__getattribute__("args")
            args[arg_name] = value
            super().__setattr__("args", args)
            args_call_count = super().__getattribute__("args_call_count")

            if arg_name in args_call_count:
                super().__setattr__("args_call_count", args_call_count)

            else:
                args_call_count[arg_name] = 0
                super().__setattr__("args_call_count", args_call_count)

        except Exception as e:
            super().__setattr__(arg_name, value)

    def get(self, key, default=None):
        
        if key in self.args_call_count:
            self.args_call_count[key] += 1
        return self.args.get(key, default)

    def update(self, *args, **kwargs):
        
        self.args.update(*args, **kwargs)
        config_check(self.args)

    def pop(self, *args):
        
        return self.args.pop(*args)

    def keys(self):
        
        return self.args.keys()

    def values(self):
        
        return self.args.values()

    def items(self):
        
        return self.args.items()

    def __str__(self):
        
        return str(self.args)

    def __repr__(self):
        
        return repr(self.args)

    def __len__(self):
        
        return len(self.args)

    def __iter__(self):
        
        return iter(self.args)

    def __contains__(self, item):
        
        return item in self.args

    def __getitem__(self, item):
        
        return self.args[item]

    def __setitem__(self, key, value):
        
        self.args[key] = value
        self.args_call_count[key] = 0
        config_check(self.args)

    def __delitem__(self, key):
        
        del self.args[key]
        config_check(self.args)

    def __eq__(self, other):
        
        return self.args == other

    def __ne__(self, other):
        
        return self.args != other
