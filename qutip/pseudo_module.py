import numpy

class PseudoModule:
    frozen = False
    def __init__(self, base_module, name=None, sub_modules=(), calls=None):
        self.base = base_module
        self.name = name or self.base.__name__

        if calls is None:
            self.calls = {}
        else:
            self.calls = calls
        self.sub_modules = {
            sub: PseudoModule(
                getattr(base_module, sub),
                name=f"{self.name}.{sub}",
                calls=self.calls
            )
            for sub in sub_modules
        }
        self.frozen = True

    def __getattr__(self, item):
        if item in self.sub_modules:
            return self.sub_modules[item]
        if f"{self.name}.{item}" in self.calls:
            self.calls[f"{self.name}.{item}"] += 1
        else:
            self.calls[f"{self.name}.{item}"] = 1
        return getattr(self.base, item)

    def __setattr__(self, item, val):
        if getattr(self, "frozen", False):
            raise(TypeError)
        super().__setattr__(item, val)

    def print(self):
        for key, val in pseudoNumpy.calls.items():
            print(f"{key:20}: {val:5}")

pseudoNumpy = PseudoModule(numpy, name="np", sub_modules=("linalg", "random"))
