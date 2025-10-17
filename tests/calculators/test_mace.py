import pytest
import torch
from neural_optimiser import test_dir
from torch_geometric.data import Data


def test_calculate_returns_expected_shapes(mace_calculator, batch):
    """Test that calculate returns tensors of expected shapes and dtypes."""
    e, f = mace_calculator(batch)

    assert isinstance(e, torch.Tensor)
    assert isinstance(f, torch.Tensor)
    assert e.shape == (batch.n_conformers,)
    assert f.shape == batch.pos.shape
    assert f.dtype == batch.pos.dtype


def test_to_atomic_data_builds_expected_fields_and_dtypes(mace_calculator, batch):
    """Test conversion to atomic Data object with expected fields and dtypes."""
    atomic = mace_calculator.to_atomic_data(batch)

    # positions concatenated
    assert hasattr(atomic, "positions")
    assert atomic.positions.shape == batch.pos.shape
    assert torch.allclose(atomic.positions, batch.pos)

    # # node features one-hot per atom with Z classes
    # assert hasattr(atomic, "node_attrs")
    # assert getattr(atomic, "node_attrs").shape[0] == batch.pos.shape[0]
    # assert getattr(atomic, "node_attrs").shape[1] == len(mace_calculator._z_table)

    # # dtype of node features matches model parameter dtype
    # model_dtype = next(iter(mace_calculator.model.parameters())).dtype
    # assert atomic.node_attrs.dtype == model_dtype

    # head per-graph index tensor
    assert hasattr(atomic, "head")
    assert atomic.head.dtype == torch.long
    assert atomic.head.numel() == batch.n_conformers

    # edge_index present (we patched to empty)
    assert hasattr(atomic, "edge_index")
    assert atomic.edge_index.shape[0] == 2


def test_atomic_numbers_mapping_and_one_hot(mace_calculator):
    """Test atomic number to index mapping and one-hot encoding."""
    z = torch.tensor([1, 8, 6, 1], dtype=torch.long)
    idx = mace_calculator.atomic_numbers_to_indices(z, z_table=mace_calculator._z_table)
    assert idx.tolist() == [0, 3, 1, 0]

    one_hot = mace_calculator.to_one_hot(idx, num_classes=len(mace_calculator._z_table))
    assert one_hot.shape == (4, len(mace_calculator._z_table))
    assert torch.all(one_hot.sum(dim=-1) == 1)


def test_validate_batch_missing_fields_raises(mace_calculator):
    """Test that _validate_batch raises errors for bad inputs."""
    bad = Data(pos=torch.zeros((1, 3)), batch=torch.zeros(1, dtype=torch.long))
    with pytest.raises(AttributeError, match="Batch must have attributes"):
        mace_calculator._validate_batch(bad)


def test_get_energies_matches_ase(mace_calculator, batch, atoms):
    """Compare energies to ASE MACE for the same model."""
    pytest.importorskip("mace", reason="MACE not installed")
    from mace.calculators.mace import MACECalculator as MACECalc

    model_path = test_dir / "models" / "MACE_SPICE2_NEUTRAL.model"
    e = mace_calculator.get_energies(batch)

    mace_calc = MACECalc(model_paths=str(model_path), device="cpu")
    atoms.calc = mace_calc
    ref_e = atoms.get_potential_energy()

    assert torch.isclose(e.squeeze(), torch.tensor(ref_e, dtype=e.dtype), atol=1e-4)


def test_mace_calculator(mace_calculator, batch, atoms):
    """Compare MACECalculator results to ASE MACE calculator."""
    pytest.importorskip("mace", reason="MACE not installed")
    from mace.calculators.mace import MACECalculator as MACECalc

    model_paths = test_dir / "models" / "MACE_SPICE2_NEUTRAL.model"
    e, f = mace_calculator(batch)

    mace_calc = MACECalc(model_paths=str(model_paths), device="cpu")
    atoms.calc = mace_calc
    _e = atoms.get_potential_energy()
    _f = atoms.get_forces()

    # Ensure comparable shapes/dtypes
    assert torch.isclose(e.squeeze(), torch.tensor(_e, dtype=e.dtype), atol=1e-4)
    assert torch.allclose(f, torch.tensor(_f, dtype=f.dtype), atol=1e-4)


def test_mace_calculator2(mace_calculator, batch, atoms):
    """Compare MACECalculator results to ASE MACE calculator."""
    pytest.importorskip("mace", reason="MACE not installed")
    from mace.calculators.mace import MACECalculator as MACECalc

    model_paths = test_dir / "models" / "MACE_SPICE2_NEUTRAL.model"
    e = mace_calculator.get_energies(batch)

    mace_calc = MACECalc(model_paths=str(model_paths), device="cpu")
    atoms.calc = mace_calc
    _e = atoms.get_potential_energy()

    # Ensure comparable shapes/dtypes
    assert torch.isclose(e.squeeze(), torch.tensor(_e, dtype=e.dtype), atol=1e-4)
