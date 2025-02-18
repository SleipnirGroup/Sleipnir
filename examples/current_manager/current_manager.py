from jormungandr.autodiff import Variable, VariableMatrix
from jormungandr.optimization import OptimizationProblem


class CurrentManager:
    """
    This class computes the optimal current allocation for a list of subsystems
    given a list of their desired currents and current tolerances that determine
    which subsystem gets less current if the current budget is exceeded.
    Subsystems with a smaller tolerance are given higher priority.
    """

    def __init__(self, current_tolerances: list[float], max_current: float):
        """
        Constructs a CurrentManager.

        Parameter ``current_tolerances``:
            The relative current tolerance of each subsystem.

        Parameter ``max_current``:
            The current budget to allocate between subsystems.
        """
        self.__desired_currents = VariableMatrix(len(current_tolerances), 1)
        self.__problem = OptimizationProblem()
        self.__allocated_currents = self.__problem.decision_variable(
            len(current_tolerances)
        )

        # Ensure desired_currents contains initialized Variables
        for row in range(self.__desired_currents.rows()):
            # Don't initialize to 0 or 1, because those will get folded by
            # Sleipnir
            self.__desired_currents[row] = Variable(float("inf"))

        J = 0.0
        current_sum = 0.0
        for i in range(len(current_tolerances)):
            # The weight is 1/tolᵢ² where tolᵢ is the tolerance between the
            # desired and allocated current for subsystem i
            error = self.__desired_currents[i] - self.__allocated_currents[i]
            J += error * error / (current_tolerances[i] * current_tolerances[i])

            current_sum += self.__allocated_currents[i]

            # Currents must be nonnegative
            self.__problem.subject_to(self.__allocated_currents[i] >= 0.0)
        self.__problem.minimize(J)

        # Keep total current below maximum
        self.__problem.subject_to(current_sum <= max_current)

    def calculate(self, desired_currents: list[float]) -> list[float]:
        """
        Returns the optimal current allocation for a list of subsystems given a
        list of their desired currents and current tolerances that determine
        which subsystem gets less current if the current budget is exceeded.
        Subsystems with a smaller tolerance are given higher priority.

        Parameter ``desired_currents``:
            The desired current for each subsystem.

        Raises ``ValueError``:
            if the number of desired currents doesn't equal the number of
            tolerances passed in the constructor.
        """
        if self.__desired_currents.rows() != len(desired_currents):
            raise ValueError(
                "Number of desired currents must equal the number of tolerances passed in the constructor."
            )

        for i in range(len(desired_currents)):
            self.__desired_currents[i].set_value(desired_currents[i])

        self.__problem.solve()

        result = []
        for i in range(len(desired_currents)):
            result.append(max(self.__allocated_currents.value(i), 0.0))

        return result
