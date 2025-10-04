import torch
from loguru import logger

from ase_bfgs_batched.optimise import Optimiser


class BFGSBatched(Optimiser):
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

        logger.info("Running BFGS optimiser")

    def step(self, forces: torch.Tensor) -> None:
        """Perform one BFGS update for each unconverged conformer.

        Skips conformers already marked as converged.
        """
        # TODO: update so this is batched if possible
        for idx, (start, end) in self._iter_conformer_slices():
            if hasattr(self.batch, "converged") and bool(self.batch.converged[idx].item()):
                logger.debug("Skipping conformer {} (already converged).", idx)
                continue

            pos_i = self.batch.pos[start:end]  # [Ni, 3]
            f_i = forces[start:end]  # [Ni, 3]

            # Flatten to 3Ni for BFGS math and use float64 for stability
            r = pos_i.reshape(-1).to(torch.float64)
            f = f_i.reshape(-1).to(torch.float64)

            d = r.numel()
            H = self._H.get(idx)
            r0 = self._r0.get(idx)
            f0 = self._f0.get(idx)

            if H is None:
                H = torch.eye(d, dtype=torch.float64, device=self.device) * 70.0
                logger.debug("Init Hessian for conformer {} with shape {}.", idx, tuple(H.shape))
            else:
                # Update Hessian using a BFGS-like rule
                dr = r - r0
                if dr.abs().max() >= 1e-7:
                    df = f - f0
                    a = torch.dot(dr, df)
                    dg = H @ dr
                    b = torch.dot(dr, dg)

                    # Safeguard denominators
                    eps = 1e-12
                    if torch.abs(a) <= eps:
                        a = (
                            torch.sign(a) * eps
                            if a != 0
                            else torch.tensor(eps, dtype=torch.float64, device=self.device)
                        )
                    if torch.abs(b) <= eps:
                        b = (
                            torch.sign(b) * eps
                            if b != 0
                            else torch.tensor(eps, dtype=torch.float64, device=self.device)
                        )

                    H = H - torch.outer(df, df) / a - torch.outer(dg, dg) / b
                    logger.debug("Updated Hessian for conformer {}.", idx)
                else:
                    logger.debug(
                        "Small displacement for conformer {}; skipping Hessian update.", idx
                    )

            # Determine step: dr = V @ ((f @ V) / |omega|)
            omega, V = torch.linalg.eigh(H)
            denom = torch.clamp(torch.abs(omega), min=1e-12)
            fV = f @ V  # [d]
            dr_flat = V @ (fV / denom)  # [d]
            dr = dr_flat.view(-1, 3).to(self.dtype)

            # Apply max step scaling
            steplengths = torch.linalg.vector_norm(dr, dim=1)  # [Ni]
            maxsteplength = torch.max(steplengths)
            if maxsteplength >= self.max_step:
                scale = (self.max_step / (maxsteplength + 1e-12)).to(self.dtype)
                dr = dr * scale
                logger.debug(
                    "Scaled step for conformer {} by factor {:.6f} (max steplength {:.6f}).",
                    idx,
                    float(scale.item()),
                    float(maxsteplength.item()),
                )

            # Update positions and carry state
            pos_i.add_(dr)
            self._H[idx] = H
            self._r0[idx] = r.detach().clone()
            self._f0[idx] = f.detach().clone()
            logger.debug("Applied step to conformer {}.", idx)


if __name__ == "__main__":
    from ase.build import molecule

    from ase_bfgs_batched.conformers import ConformerBatch

    atoms_list = [molecule("H2O"), molecule("NH3"), molecule("CH4")]
    batch = ConformerBatch.from_ase(atoms_list * 100)

    optimiser = BFGSBatched(steps=10, fmax=0.05, fexit=500.0)
    optimiser.calculator = 0  # Replace with actual calculator implementing forces(batch)
    converged = optimiser.run(batch)
