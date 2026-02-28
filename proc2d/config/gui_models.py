"""Typed model for GUI run parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from ..mask import openings_from_any


@dataclass(slots=True)
class GuiRunConfig:
    """Normalized GUI inputs used to build a simulation deck."""

    Lx_um: float = 2.0
    Ly_um: float = 0.5
    Nx: int = 401
    Ny: int = 201
    background_doping_cm3: float = 1.0e15

    openings_um: list[list[float]] = field(default_factory=lambda: [[0.8, 1.2]])
    sigma_lat_um: float = 0.03

    dose_cm2: float = 1.0e13
    Rp_um: float = 0.05
    dRp_um: float = 0.02

    oxidation_enable: bool = False
    oxidation_time_s: float = 5.0
    oxidation_A_um: float = 0.1
    oxidation_B_um2_s: float = 0.01
    oxidation_gamma: float = 2.27
    oxidation_apply_on: str = "all"
    oxidation_consume_dopants: bool = True
    oxidation_update_materials: bool = True

    D_cm2_s: float = 1.0e-14
    total_t_s: float = 10.0
    dt_s: float = 0.5
    anneal_use_arrhenius: bool = False
    arrhenius_D0_cm2_s: float = 1.0e-3
    arrhenius_Ea_eV: float = 3.5
    arrhenius_T_C: float = 1000.0
    arrhenius_schedule: list[dict[str, float]] | None = None

    oxide_D_scale: float = 0.0
    cap_eps_um: float = 0.0

    top_open_type: str = "robin"
    h_cm_s: float = 1.0e-5
    Ceq_cm3: float = 0.0
    dirichlet_value_cm3: float = 0.0

    outdir: str = "outputs/gui_run"
    formats: list[str] = field(default_factory=lambda: ["npy", "csv", "png"])
    linecut_x_um: float = 1.0
    linecut_y_um: float = 0.05
    plot_log10: bool = True
    plot_vmin: float = 1.0e14
    plot_vmax: float = 1.0e20
    export_vtk: bool = False
    export_tox_csv: bool = True
    export_tox_png: bool = True

    compute_metrics: bool = True
    record_history: bool = False
    history_every_s: float = 0.5
    history_save_png: bool = True
    history_save_csv: bool = True

    zip_outputs: bool = False
    store_full_c: bool = False

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> GuiRunConfig:
        """Build normalized GUI config from a mapping."""
        cfg = cls()

        def _pick(name: str, default: Any) -> Any:
            return values[name] if name in values else default

        cfg.Lx_um = float(_pick("Lx_um", cfg.Lx_um))
        cfg.Ly_um = float(_pick("Ly_um", cfg.Ly_um))
        cfg.Nx = int(_pick("Nx", cfg.Nx))
        cfg.Ny = int(_pick("Ny", cfg.Ny))
        cfg.background_doping_cm3 = float(_pick("background_doping_cm3", cfg.background_doping_cm3))

        openings = _pick("openings_um", cfg.openings_um)
        cfg.openings_um = openings_from_any(openings) if openings else []
        cfg.sigma_lat_um = float(_pick("sigma_lat_um", cfg.sigma_lat_um))

        cfg.dose_cm2 = float(_pick("dose_cm2", cfg.dose_cm2))
        cfg.Rp_um = float(_pick("Rp_um", cfg.Rp_um))
        cfg.dRp_um = float(_pick("dRp_um", cfg.dRp_um))

        cfg.oxidation_enable = bool(_pick("oxidation_enable", cfg.oxidation_enable))
        cfg.oxidation_time_s = float(_pick("oxidation_time_s", cfg.oxidation_time_s))
        cfg.oxidation_A_um = float(_pick("oxidation_A_um", cfg.oxidation_A_um))
        cfg.oxidation_B_um2_s = float(_pick("oxidation_B_um2_s", cfg.oxidation_B_um2_s))
        cfg.oxidation_gamma = float(_pick("oxidation_gamma", cfg.oxidation_gamma))
        cfg.oxidation_apply_on = str(_pick("oxidation_apply_on", cfg.oxidation_apply_on))
        cfg.oxidation_consume_dopants = bool(
            _pick("oxidation_consume_dopants", cfg.oxidation_consume_dopants)
        )
        cfg.oxidation_update_materials = bool(
            _pick("oxidation_update_materials", cfg.oxidation_update_materials)
        )

        cfg.D_cm2_s = float(_pick("D_cm2_s", cfg.D_cm2_s))
        cfg.total_t_s = float(_pick("total_t_s", cfg.total_t_s))
        cfg.dt_s = float(_pick("dt_s", cfg.dt_s))
        cfg.anneal_use_arrhenius = bool(_pick("anneal_use_arrhenius", cfg.anneal_use_arrhenius))
        cfg.arrhenius_D0_cm2_s = float(_pick("arrhenius_D0_cm2_s", cfg.arrhenius_D0_cm2_s))
        cfg.arrhenius_Ea_eV = float(_pick("arrhenius_Ea_eV", cfg.arrhenius_Ea_eV))
        cfg.arrhenius_T_C = float(_pick("arrhenius_T_C", cfg.arrhenius_T_C))

        schedule = _pick("arrhenius_schedule", cfg.arrhenius_schedule)
        if schedule is None:
            cfg.arrhenius_schedule = None
        else:
            cfg.arrhenius_schedule = []
            for seg in schedule:
                if not isinstance(seg, Mapping):
                    continue
                if "t_s" not in seg or "T_C" not in seg:
                    continue
                cfg.arrhenius_schedule.append({"t_s": float(seg["t_s"]), "T_C": float(seg["T_C"])})

        cfg.oxide_D_scale = float(_pick("oxide_D_scale", cfg.oxide_D_scale))
        cfg.cap_eps_um = float(_pick("cap_eps_um", cfg.cap_eps_um))

        cfg.top_open_type = str(_pick("top_open_type", cfg.top_open_type))
        cfg.h_cm_s = float(_pick("h_cm_s", cfg.h_cm_s))
        cfg.Ceq_cm3 = float(_pick("Ceq_cm3", cfg.Ceq_cm3))
        cfg.dirichlet_value_cm3 = float(_pick("dirichlet_value_cm3", cfg.dirichlet_value_cm3))

        cfg.outdir = str(_pick("outdir", cfg.outdir))
        formats = _pick("formats", cfg.formats)
        cfg.formats = [str(fmt).lower() for fmt in formats]
        cfg.linecut_x_um = float(_pick("linecut_x_um", cfg.linecut_x_um))
        cfg.linecut_y_um = float(_pick("linecut_y_um", cfg.linecut_y_um))
        cfg.plot_log10 = bool(_pick("plot_log10", cfg.plot_log10))
        cfg.plot_vmin = float(_pick("plot_vmin", cfg.plot_vmin))
        cfg.plot_vmax = float(_pick("plot_vmax", cfg.plot_vmax))
        cfg.export_vtk = bool(_pick("export_vtk", cfg.export_vtk))
        cfg.export_tox_csv = bool(_pick("export_tox_csv", cfg.export_tox_csv))
        cfg.export_tox_png = bool(_pick("export_tox_png", cfg.export_tox_png))

        cfg.compute_metrics = bool(_pick("compute_metrics", cfg.compute_metrics))
        cfg.record_history = bool(_pick("record_history", cfg.record_history))
        cfg.history_every_s = float(_pick("history_every_s", cfg.history_every_s))
        cfg.history_save_png = bool(_pick("history_save_png", cfg.history_save_png))
        cfg.history_save_csv = bool(_pick("history_save_csv", cfg.history_save_csv))

        cfg.zip_outputs = bool(_pick("zip_outputs", cfg.zip_outputs))
        cfg.store_full_c = bool(_pick("store_full_c", cfg.store_full_c))

        return cfg

    def to_mapping(self) -> dict[str, Any]:
        """Return config as a plain dictionary."""
        return asdict(self)
