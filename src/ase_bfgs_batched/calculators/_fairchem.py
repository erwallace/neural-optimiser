class FAIRChemCalculator:
    def __init__(self, model_path):
        try:
            from fairchem import FAIRChemCalculator as FAIRChemCalc
        except ImportError:
            raise ImportError("MACE is not installed. Please install it to use MACECalculator.")
        self.calculator = FAIRChemCalc(model_path)

    def calculate(self, atoms):
        return self.calculator.calculate(atoms)

    def to_atomic_data():
        pass

    def from_atomic_data():
        pass
