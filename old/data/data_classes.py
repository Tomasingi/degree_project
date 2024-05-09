class Doctor:
    def __init__(self, name, cardiac=False, charge=False):
        self.name = name
        self.cardiac = cardiac
        self.charge = charge

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

class Operation:
    def __init__(self, id, start_time, duration, cardiac=False):
        self.id = id
        self.start_time = start_time
        self.duration = duration
        self.cardiac = cardiac

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)