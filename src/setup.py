"""
Load experiment setup

roger.bermudez@epfl.ch
CVLab EPFL 2019
"""

from datetime import datetime
import yaml


class Struct:
    '''
    Generic structure that wraps dictionary to register fields and pretty-print
    '''
    def __init__(self, **params):
        super(Struct, self).__init__()
        for key, value in params.items():
            self.__setitem__(key, value)

    def __len__(self):
        return len(self.__dict__)

    def __iter__(self):
        return iter(self.__dict__.items())

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = Struct(**value)
        self.__dict__[key] = value

    def iteritems(self):
        return self.__dict__.items()

    def iterkeys(self):
        return self.__dict__.keys()

    def itervalues(self):
        return self.__dict__.values()

    def __repr__(self):
        str_repr = []
        for key, value in self.__dict__.items():
            if key.startswith('_'):
                continue
            if isinstance(value, dict):
                value_str = repr(Struct(**value))
            elif isinstance(value, datetime):
                value_str = value.strftime('%y.%m.%d %H:%M:%S')
            else:
                value_str = repr(value)
            if value_str.find("\n") >= 0 or isinstance(value, type(self)):
                value_str = "\n" + "\n".join("   " + line for line in value_str.split("\n"))
            str_repr.append(f"{key} = {value_str}")
        return "\n".join(str_repr)


def load_experiment(experiment_path):
    '''
    Loads a yaml file with experiment description.
    '''
    def _fix_keys(experiment):
        '''
        Normalizes keys in a dictionary so that they are lowercase with no spaces.
        '''
        if isinstance(experiment, dict):
            new_experiment = {}
            for key, value in experiment.items():
                new_key = key.replace(' ', '_').lower()
                new_experiment[new_key] = _fix_keys(value)
            experiment = new_experiment
        return experiment

    experiment = yaml.safe_load(open(experiment_path, 'r'))
    experiment = _fix_keys(experiment)
    experiment = Struct(**experiment)
    return experiment
