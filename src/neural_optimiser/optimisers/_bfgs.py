import torch

from neural_optimiser.optimisers.base import Optimiser


class BFGS(Optimiser):
    def __init__(
        self,
        max_step: float = 0.04,
        steps: int = -1,
        fmax: float | None = None,
        fexit: float | None = None,
    ) -> None:
        # BFGS state (per conformer)
        self._H: dict[int, torch.Tensor] = {}
        self._r0: dict[int, torch.Tensor] = {}
        self._f0: dict[int, torch.Tensor] = {}

        super().__init__(max_step, steps, fmax, fexit)

    def _reset_batch_state(self) -> None:
        """Reset any per-batch state before starting a new batch."""
        self._H.clear()
        self._r0.clear()
        self._f0.clear()

    def run(self, batch):
        """Reset state and run optimisation on a new batch."""
        self._reset_batch_state()
        return super().run(batch)

    def step(self, forces: torch.Tensor) -> None:
        """Batched BFGS update across all conformers."""
        ptr = self.batch.ptr
        N = self.batch.n_conformers

        # Per-conformer sizes
        Ns = [(ptr[i + 1] - ptr[i]).item() for i in range(N)]  # atoms per conformer
        dis = [int(n * 3) for n in Ns]  # flattened dims
        Dmax = max(dis)
        Nmax = max(int(n) for n in Ns)

        # Pack r, f (padded)
        r = torch.zeros((N, Dmax), dtype=self.dtype, device=self.device)
        f = torch.zeros((N, Dmax), dtype=self.dtype, device=self.device)
        r0 = torch.zeros((N, Dmax), dtype=self.dtype, device=self.device)
        f0 = torch.zeros((N, Dmax), dtype=self.dtype, device=self.device)
        have_prev = torch.zeros(N, dtype=torch.bool, device=self.device)

        for i, (start, end) in self._iter_conformer_slices():
            di = dis[i]
            r[i, :di] = self.batch.pos[start:end].reshape(-1).to(self.dtype)
            f[i, :di] = forces[start:end].reshape(-1).to(self.dtype)
            if i in self._H:
                have_prev[i] = True
                r0[i, :di] = self._r0[i]
                f0[i, :di] = self._f0[i]

        # Build batched Hessian (padded)
        H = torch.zeros((N, Dmax, Dmax), dtype=self.dtype, device=self.device)
        for i in range(N):
            di = dis[i]
            if i in self._H:
                H[i, :di, :di] = self._H[i]
            else:
                H[i, :di, :di] = torch.eye(di, dtype=self.dtype, device=self.device) * 70.0

        # BFGS update (masked to active dims; skipped for first-iteration conformers)
        dr = r - r0
        df = f - f0
        idxs = torch.arange(Dmax, device=self.device).unsqueeze(0).expand(N, -1)
        active_mask = idxs < torch.tensor(dis, device=self.device).unsqueeze(1)
        dr = dr * active_mask
        df = df * active_mask

        dg = torch.bmm(H, dr.unsqueeze(-1)).squeeze(-1)

        eps = torch.tensor(1e-12, dtype=self.dtype, device=self.device)

        def safe_den(x: torch.Tensor) -> torch.Tensor:
            sign = torch.sign(x)
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            return torch.where(torch.abs(x) <= eps, sign * eps, x)

        a = safe_den((dr * df).sum(dim=-1))  # [B]
        b = safe_den((dr * dg).sum(dim=-1))  # [B]

        df_outer = df.unsqueeze(-1) * df.unsqueeze(-2)  # [B, Dmax, Dmax]
        dg_outer = dg.unsqueeze(-1) * dg.unsqueeze(-2)
        upd = -df_outer / a.view(N, 1, 1) - dg_outer / b.view(N, 1, 1)
        H = H + upd * have_prev.view(N, 1, 1)  # no update on first iteration

        # Batched eigendecomposition and step
        omega, V = torch.linalg.eigh(H)  # [B, Dmax], [B, Dmax, Dmax]
        denom = torch.clamp(torch.abs(omega), min=1e-12)
        fV = torch.bmm(f.unsqueeze(1), V).squeeze(1)  # [B, Dmax]
        dr_flat = torch.bmm(V, (fV / denom).unsqueeze(-1)).squeeze(-1).to(self.dtype)  # [B, Dmax]

        # Per-conformer max_step scaling (padded)
        dr_atoms = torch.zeros((N, Nmax, 3), dtype=self.dtype, device=self.device)
        valid_mask = torch.zeros((N, Nmax), dtype=torch.bool, device=self.device)
        for i in range(N):
            Ni, di = int(Ns[i]), dis[i]
            if Ni > 0:
                dr_atoms[i, :Ni, :] = dr_flat[i, :di].view(Ni, 3)
                valid_mask[i, :Ni] = True

        steplengths = torch.linalg.vector_norm(dr_atoms, dim=2)  # [B, Nmax]
        maxsteplength = (
            torch.where(
                valid_mask,
                steplengths,
                torch.tensor(float("-inf"), device=self.device, dtype=steplengths.dtype),
            )
            .max(dim=1)
            .values
        )
        scales = (self.max_step / (maxsteplength + 1e-12)).clamp(max=1.0).to(self.dtype)  # [B]
        dr_atoms = dr_atoms * scales.view(N, 1, 1)

        # Apply updates and store per-conformer state
        for i, (start, end) in self._iter_conformer_slices():
            Ni, di = int(Ns[i]), dis[i]
            self.batch.pos[start:end].add_(dr_atoms[i, :Ni, :])
            self._H[i] = H[i, :di, :di].detach().clone()
            self._r0[i] = r[i, :di].detach().clone()
            self._f0[i] = f[i, :di].detach().clone()


if __name__ == "__main__":
    from timeit import default_timer as timer

    from ase.build import molecule

    from neural_optimiser.calculators import RandomCalculator
    from neural_optimiser.conformers import ConformerBatch

    device = "cpu"

    atoms_list = [molecule("H2O"), molecule("NH3"), molecule("CH4")] * 10
    batch = ConformerBatch.from_ase(atoms_list, device=device)

    start = timer()
    optimiser = BFGS(steps=2000, fmax=0.05, fexit=500.0)
    optimiser.calculator = RandomCalculator()
    converged = optimiser.run(batch)
    end = timer()

    print(f"Single BFGS call took {end - start:.2f} seconds.")

    # 30 molecules for 2000 steps with a dummy calculator: ~3.3 seconds

    from neural_optimiser.calculators import MACECalculator
    from neural_optimiser.conformers import ConformerBatch

    device = "cpu"

    atoms_list = [molecule("H2O"), molecule("NH3"), molecule("CH4")]
    batch = ConformerBatch.from_ase(atoms_list, device=device)

    start = timer()
    optimiser = BFGS(steps=20, fmax=0.05, fexit=500.0)
    optimiser.calculator = MACECalculator(model_paths="./models/MACE_SPICE2_NEUTRAL.model")
    converged = optimiser.run(batch)
    end = timer()

    print(f"Single BFGS call took {end - start:.2f} seconds.")
