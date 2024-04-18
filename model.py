import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp

from data.data_classes import Doctor, Operation
from data.simulator import sample_day

class MiniAllocationData:
    def __init__(self):
        self.operations = sample_day()

        self.cardiac_operations = [
            s for s in self.operations if s.cardiac
        ]

        self.doctors = [
            Doctor('Steve'),
            Doctor('Steve 2'),
            Doctor('Bob'),
            Doctor('Joey'),
            Doctor('Carl'),
            Doctor('Steve 3', cardiac=True),
            Doctor('Zarathustra', cardiac=True),
            Doctor('Bill', charge=True),
            Doctor('Billybob', cardiac=True, charge=True),
        ]

        self.cardiac_doctors = [
            d for d in self.doctors if d.cardiac
        ]

        self.charge = [
            d for d in self.doctors if d.charge
        ]

class MiniAllocationModel:
    def __init__(self):
        self.D = MiniAllocationData()

        self.M = gp.Model('Mini model')
        self.x = self.M.addVars([(d,op) for d in self.D.doctors for op in self.D.operations], vtype=gp.GRB.BINARY)

        self.w = self.M.addVars([op for op in self.D.operations], lb=0.0, vtype=gp.GRB.CONTINUOUS)
        # self.y = self.M.addVars([
        #     (op1,op2) for op1 in self.D.operations for op2 in self.D.operations
        #     if op1.id < op2.id
        #     ], vtype=gp.GRB.BINARY)

        # self.l = self.M.addVars([d for d in self.D.doctors], vtype=gp.GRB.CONTINUOUS)

        self.BIGM = 1e3

        self.weight_delay = 1e3
        self.weight_leave = 1

        self.M.setObjective(
            gp.quicksum(
                self.w[op]
                for op in self.D.operations
            ) * self.weight_delay

            # + gp.quicksum(
            #     self.l[d]
            #     for d in self.D.doctors
            # ) * self.weight_leave
            , gp.GRB.MINIMIZE
        )

        # One doctor per operation
        self.M.addConstrs(
            gp.quicksum(
                self.x[d,op] for d in self.D.doctors
            ) == 1
            for op in self.D.operations
        )

        # One cardiac doctor per cardiac operation
        self.M.addConstrs(
            gp.quicksum(
                self.x[d,op] for d in self.D.cardiac_doctors
            ) == 1
            for op in self.D.cardiac_operations
        )

        # One operation per doctor at a time
        # self.M.addConstrs(
        #     self.x[d,op1] + self.x[d,op2] <= 1
        #     for d in self.D.doctors
        #     for op1 in self.D.operations
        #     for op2 in self.D.operations
        #     if op1 != op2
        #     and (op1.start_time + op1.duration >= op2.start_time)
        #     and (op2.start_time + op2.duration >= op1.start_time)
        # )
        self.M.addConstrs(
            op1.start_time + op1.duration + self.w[op1]
            <= op2.start_time + self.w[op2]
            + self.BIGM * (2 - self.x[d,op1] - self.x[d,op2])
            for d in self.D.doctors
            for op1 in self.D.operations
            for op2 in self.D.operations
            if op1.start_time < op2.start_time
        )

        self.M.addConstrs(
            op1.start_time + op1.duration + self.w[op1]
            <= op2.start_time + self.w[op2]
            + self.BIGM * (2 - self.x[d,op1] - self.x[d,op2])
            for d in self.D.doctors
            for op1 in self.D.operations
            for op2 in self.D.operations
            if op1.start_time == op2.start_time
            and op1.id < op2.id
        )

        # self.M.addConstrs(
        #     op1.start_time + op1.duration + self.w[op1]
        #     <= op2.start_time + self.w[op2]
        #     + self.BIGM * (2 - self.x[d,op1] - self.x[d,op2])
        #     + self.BIGM * self.y[op1,op2]
        #     for d in self.D.doctors
        #     for op1 in self.D.operations
        #     for op2 in self.D.operations
        #     if op1.id < op2.id
        # )

        # self.M.addConstrs(
        #     op2.start_time + op2.duration + self.w[op2]
        #     <= op1.start_time + self.w[op1]
        #     + self.BIGM * (2 - self.x[d,op1] - self.x[d,op2])
        #     + self.BIGM * (1 - self.y[op1,op2])
        #     for d in self.D.doctors
        #     for op1 in self.D.operations
        #     for op2 in self.D.operations
        #     if op1.id < op2.id
        # )

        # Start of the day
        # self.M.addConstrs(
        #     self.l[d] >= 7
        #     for d in self.D.doctors
        # )

        # Doctors leave only after all assigned operations
        # self.M.addConstrs(
        #     self.l[d] >= (op.start_time + op.duration + self.w[op]) - self.BIGM * (1 - self.x[d,op])
        #     for d in self.D.doctors
        #     for op in self.D.operations
        # )

        # Doctors leave in the assigned order
        # self.M.addConstrs(
        #     self.l[self.D.doctors[i]] <= self.l[self.D.doctors[i+1]]
        #     for i in range(len(self.D.doctors) - 1)
        # )

    def print_operations(self):
        for op in self.D.operations:
            print(f'{op.id:<3}{"C" if op.cardiac else "-"} {op.start_time} {op.duration}')

    def solve(self, verbose=False):
        self.M.setParam('OutputFlag', 0)
        if verbose:
            self.M.setParam('OutputFlag', 1)
        self.M.optimize()

    def print_solution(self):
        print()
        print("=== Delays ===")
        for op in self.D.operations:
            print(f'{op.id:<3}{round(self.w[op].X, 2)}')

        print()
        print("=== Leaving times ===")
        for d in self.D.doctors:
            print(f'{d.name:15}{round(self.l[d].X, 2)}')

        print()
        print("=== Operation allocations ===")
        for op in self.D.operations:
            for d in self.D.doctors:
                if (self.x[d,op].X > 0.5):
                    print(f'{op.id:<3}{d.name:15}')

    def plot_allocations(self):
        fig, ax = plt.subplots()

        doctor_ids = dict()
        for i, d in enumerate(self.D.doctors):
            doctor_ids[d.name] = i

        for d in self.D.doctors:
            for op in self.D.operations:
                if (self.x[d,op].X > 0.5):
                    ax.plot([op.start_time + self.w[op].X, op.start_time + self.w[op].X + op.duration], [doctor_ids[d.name], doctor_ids[d.name]], 'k')

        ax.set_yticks(range(len(self.D.doctors)))
        ax.set_yticklabels([d.name for d in self.D.doctors])

        plt.show()

def main():
    pass

if __name__ == '__main__':
    main()