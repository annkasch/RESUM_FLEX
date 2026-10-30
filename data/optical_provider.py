"""Dataset provider for prepared LAr optical-map data (a view on the shared event cache).

Configs select it with ``data_provider: <package>.provider:OpticalDataset`` and may set
``data_config`` to an optical data-preparation YAML.
"""

from pathlib import Path

from data.optical_view import OpticalView, voxel_centers


class OpticalDataset:
    def __init__(self, root):
        self.root = Path(root)
        self._view = None

    @property
    def view(self) -> OpticalView:
        if self._view is None:
            if not (self.root / "view.json").is_file():
                raise FileNotFoundError(f"No prepared optical data (view.json) in {self.root}")
            self._view = OpticalView(self.root)
        return self._view

    def partitions(self):
        return list(self.view.partitions)

    def load(self, partition, fidelity, *, labels=True):
        view = self.view
        batch = (
            view.load(partition, fidelity) if labels else view.load_inputs(partition, fidelity)[0]
        )
        return batch, view.provenance(partition, fidelity)

    def theta(self, partition, fidelity):
        return self.view.load_theta(partition, fidelity)

    def coordinates(self, partition, fidelity):
        """Voxel centers [B, 3] in m (filename or observed midrange, as configured)."""
        view = self.view
        selected = view.select(partition, fidelity)
        rows = view.cache.locate(view.paths(selected), selected["sha256"])
        return voxel_centers(view.cache, rows, view.config)

    def normalization(self):
        return self.view.meta["normalization"]

    @classmethod
    def prepare(cls, data_config, data_directory):
        """Rebuild ``data_directory`` from the optical data config if outdated; returns a record."""
        from data.optical_pipeline import ensure_prepared_optical_data
        from schemas.optical import load_optical_config

        preparation = load_optical_config(data_config)
        data_directory = Path(data_directory).resolve()
        if preparation.output_directory != data_directory:
            raise ValueError(
                f"data_config writes to {preparation.output_directory}, "
                f"but data_directory is {data_directory}"
            )
        # config.json is written only by a full preparation.
        saved = data_directory / "config.json"
        before = saved.stat().st_mtime_ns if saved.exists() else None
        ensure_prepared_optical_data(preparation)
        regenerated = before != saved.stat().st_mtime_ns
        print(
            f"Prepared data {'regenerated' if regenerated else 'up to date'}: {data_directory}",
            flush=True,
        )
        return dict(
            path=str(data_config),
            regenerated=regenerated,
            settings=preparation.model_dump(mode="json"),
        )
