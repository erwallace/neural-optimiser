class MACECalculator():
    def __init__(self, model_path):
        try:
            from mace import MACECalculator as MACECalc
        except ImportError:
            raise ImportError("MACE is not installed. Please install it to use MACECalculator.")
        self.calculator = MACECalc(model_path)

    def calculate(self, atoms):
        return self.calculator.calculate(atoms)
    
    def to_atomic_data():
        pass

    def from_atomic_data():
        pass