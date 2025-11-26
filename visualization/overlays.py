import torch, numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
from typing import Tuple, Sequence, List, Union, Optional, Iterator
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim

from utils.ct_views import get_ct_view, view_to_axis
from utils.dvf import get_dvf_components, warp_image


class DVFOverlay:
    """
    Render DVF arrows over CT slices and save to PNGs.
    Arrow color encodes true in-plane displacement magnitude.
    """

    MAG_RANGE = (0.01, 0.08)
    QUIVER_KW = dict(
        angles="xy",
        scale_units="xy",
        headwidth=3,
        headlength=3,
        headaxislength=3.4,
        pivot="tail",
    )

    def __init__(
        self,
        max_arrows: int = 1600,
        dvf_cmap: str = "plasma",
        diffmap_cmap: str = "magma",
        alpha: float = 1.0,
        scale: float = 0.01,
        width: float = 0.003,
        heatmap_sigma: float = 0.5,
    ) -> None:
        """
        Args:
            max_arrows (int): Maximum number of arrows to overlay on the image.
            cmap (str): Color map to use; defaults to blue -> red.
            alpha (float): The opacity of the arrows. 0 = fully transparent, 1 = fully opaque.
            scale (float): The scale of the arrows.
            width (float): The width of the arrows.
            heatmap_sigma (float): The sigma of the gaussian blur applied to the heatmap.
        """

        self.max_arrows = max_arrows
        self.dvf_cmap = dvf_cmap
        self.diffmap_cmap = diffmap_cmap
        self.alpha = alpha
        self.scale = scale
        self.width = width
        self.heatmap_sigma = heatmap_sigma

    def _discretize_grid(self, grid_rows: np.ndarray, grid_cols: np.ndarray):
        """Make a grid of arrow coordinates such that the total number of arrows is <= max_arrows."""
        n_rows, n_cols = grid_rows.shape
        n_points = n_rows * n_cols
        if n_points > self.max_arrows:
            stride = int(np.ceil(np.sqrt(n_points / self.max_arrows)))
            discretized_rows = grid_rows[::stride, ::stride]
            discretized_cols = grid_cols[::stride, ::stride]
        else:
            discretized_rows = grid_rows
            discretized_cols = grid_cols
        return discretized_rows, discretized_cols

    def _make_grid(
        self,
        img_slice: np.ndarray,
        axis: int,
        spacing_mm: Optional[Tuple[float, float, float]] = None,
    ) -> tuple[np.ndarray, np.ndarray, Optional[tuple[float, float, float, float]]]:
        """
        Build row/col coordinate grids (in pixels or mm) for a slice plane.

        Args:
            img_slice (np.ndarray): 2D slice from the CT volume
            axis: 0=axial(rows=y, cols=x), 1=coronal(rows=z, cols=x), 2=sagittal(rows=z, cols=y)
            spacing_mm: (dz, dy, dx); when None, coordinates are pixel indices and extent=None.

        Returns:
            Tuple[np.ndarray, np.ndarray, Optional[tuple[float, float, float, float]]]:
                - rows: Row coordinates (in pixels or mm)
                - cols: Column coordinates (in pixels or mm)
                - extent: Tuple of (left, right, top, bottom) in mm, or None if spacing_mm is None
        """

        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1, or2, got {axis}")

        n_rows, n_cols = img_slice.shape
        rows, cols = np.mgrid[0:n_rows, 0:n_cols]

        if spacing_mm is None:
            return rows, cols, None

        dz, dy, dx = spacing_mm
        if axis == 0:  # axial
            row_spacing, col_spacing = dy, dx
        elif axis == 1:  # coronal
            row_spacing, col_spacing = dz, dx
        else:  # sagittal
            row_spacing, col_spacing = dz, dy

        rows = rows * row_spacing
        cols = cols * col_spacing
        extent = (0.0, n_cols * col_spacing, 0, n_rows * row_spacing)
        return rows, cols, extent

    def _slice_indices(self, size: int, num_slices: int, start_from: str) -> List[int]:
        """Get the indices of the slices to render for a certain view (axis)."""
        if num_slices < 1:
            return []
        s = start_from.lower()
        if s == "middle":
            start = max(0, (size // 2) - (num_slices // 2))
        elif s == "first":
            start = 0
        elif s == "last":
            start = max(0, size - num_slices)
        else:
            raise ValueError(
                f"start_from must be 'middle', 'first', or 'last', got {start_from!r}"
            )
        stop = min(size, start + num_slices)
        return list(range(start, stop))

    def _iter_slices(
        self, vol: torch.Tensor, views: Sequence[str], num_slices: int, start_from: str
    ) -> Iterator[Tuple[str, int, int]]:
        """Helper to iterate over slices in the volume."""
        for view in views:
            axis = view_to_axis(view)
            for idx in self._slice_indices(vol.shape[axis], num_slices, start_from):
                yield view, axis, idx

    def _overlay_dvf(
        self,
        ax: plt.Axes,
        mode: str,
        mag: np.ndarray,
        rows: np.ndarray | None = None,
        cols: np.ndarray | None = None,
        disp_rows: np.ndarray | None = None,
        disp_cols: np.ndarray | None = None,
        extent: tuple[float, float, float, float] | None = None,
        origin: str = "lower",
    ):
        """
        Draw either arrows or a heatmap onto *ax* and return the mappable.
        Raises for an unknown mode.
        """
        vmin, vmax = self.MAG_RANGE

        if mode == "arrows":
            return ax.quiver(
                cols,
                rows,
                disp_cols,
                disp_rows,
                mag,
                cmap=self.dvf_cmap,
                clim=self.MAG_RANGE,
                scale=self.scale,
                width=self.width,
                alpha=self.alpha,
                **self.QUIVER_KW,
            )

        if mode == "heatmap":
            return ax.imshow(
                mag,
                cmap=self.dvf_cmap,
                vmin=vmin,
                vmax=vmax,
                alpha=self.alpha,
                origin=origin,
                extent=extent,
            )

        raise ValueError(f"mode must be 'arrows' or 'heatmap', got {mode!r}")

    def _render_slice_with_overlay(
        self,
        out_dir: Path,
        img: np.ndarray,
        grid_rows: np.ndarray,
        grid_cols: np.ndarray,
        disp_rows: np.ndarray,
        disp_cols: np.ndarray,
        view: str,
        idx: int,
        extent: tuple[float, float, float, float] | None,
        mag: np.ndarray,
        mode: str = "arrows",
    ) -> Path:
        """Render a single slice and return the saved PNG path.

        Args:
            out_dir (Path): The directory to save the PNGs.
            img (np.ndarray): The image to render.
            grid_rows (np.ndarray): The row coordinates of the grid.
            grid_cols (np.ndarray): The column coordinates of the grid.
            disp_rows (np.ndarray): The row (height-wise) displacements.
            disp_cols (np.ndarray): The column (width-wise) displacements.
            view (str): The view to render.
            idx (int): The index of the slice to render.
            extent (tuple[float, float, float, float] | None): The physical extent (size) of the image.
            spacing_mm (tuple[float, float, float] | None): The spacing of the volume in mm (dz, dy, dx).
            mag (np.ndarray): The magnitude of the displacement.
            mode (str): The mode to render the overlay in. Can be "arrows" or "heatmap".
        """
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

        origin = "lower"
        if view == "axial":
            extent = (extent[0], extent[1], extent[3], extent[2])  # flip top-bottom
            origin = "upper"
        ax.imshow(
            img,
            cmap="gray",
            vmin=-1,
            vmax=1,
            origin=origin,
            interpolation="nearest",
            extent=extent,
        )

        overlay = self._overlay_dvf(
            ax,
            mode=mode,
            mag=mag,
            rows=grid_rows,
            cols=grid_cols,
            disp_rows=disp_rows,
            disp_cols=disp_cols,
            extent=extent,
            origin=origin,
        )

        ax.set_axis_off()

        fname = out_dir / f"{view}_idx{idx:04d}_overlay.png"
        fig.savefig(fname, bbox_inches=None, pad_inches=0)
        plt.close(fig)
        return fname

    def render_dvf_overlay(
        self,
        out_dir: Union[str, Path],
        vol: torch.Tensor,
        dvf: torch.Tensor,
        views: Sequence[str] = ("axial", "coronal", "sagittal"),
        num_slices: int = 1,
        start_from: str = "middle",
        spacing_mm: Tuple[float, float, float] = None,
        use_arrows: bool = True,
    ):
        """
        Render DVF arrows over CT slices and save to PNGs.
        Arrow color encodes true in-plane displacement magnitude.

        Args:
            vol (torch.Tensor): CT volume of shape [D,H,W]
            dvf (torch.Tensor): DVF of shape [3,D,H,W] where channels correspond to the z,y,x displacements
            views (Sequence[str]): Views to render. Can be "axial", "coronal", or "sagittal".
            num_slices (int): Number of slices to render per view. If 1, render the middle slice.
            start_from (str): "middle", "last", or "first".
                If "middle", render the middle slice + num_slices//2 slices before and after.
                If "last", render the last slice - num_slices//2 slices before.
                If "first", render the first slice + num_slices//2 slices after.
            spacing_mm (Tuple[float,float,float]): Spacing of the volume in mm.
            percentile (float): The percentile to limit plotting to. Will cut off top and bottom (percentile)% values of the distribution.
            use_arrows (bool): Whether to use arrows or heatmap. If heatmap is used, a gaussian blur is applied to the magnitude.
        Returns:
            List[Path]: Paths to the saved PNGs.
        """

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        vol_np = vol.detach().cpu().numpy()  # [D,H,W]
        dvf_np = dvf.detach().cpu().numpy()  # [3,D,H,W]

        dvf_np = [dvf_np[i] * spacing_mm[i] for i in range(len(spacing_mm))]
        dvf_np = np.stack(dvf_np)

        saved = []

        for view, axis, idx in self._iter_slices(vol, views, num_slices, start_from):
            slice = get_ct_view(vol_np, view=view, idx=idx)
            displacement_rows, displacement_cols = get_dvf_components(
                dvf_np, view=view, idx=idx
            )

            grid_rows, grid_cols, extent = self._make_grid(slice, axis, spacing_mm)

            if use_arrows:
                grid_rows, grid_cols = self._discretize_grid(grid_rows, grid_cols)
                displacement_rows, displacement_cols = self._discretize_grid(
                    displacement_rows, displacement_cols
                )

            mode = "arrows" if use_arrows else "heatmap"

            mag = np.hypot(displacement_rows, displacement_cols)

            if mode == "heatmap":
                mag = gaussian_filter(mag, sigma=self.heatmap_sigma)

            fname = self._render_slice_with_overlay(
                out_dir,
                slice,
                grid_rows,
                grid_cols,
                displacement_rows,
                displacement_cols,
                view,
                idx,
                extent,
                mag,
                mode,
            )
            saved.append(fname)
        return saved

    def render_difference_map(
        self,
        out_dir: Union[str, Path],
        fixed_vol: torch.Tensor,
        moving_vol: torch.Tensor,
        dvf: torch.Tensor,
        views: Sequence[str] = ("axial", "coronal", "sagittal"),
        num_slices: int = 1,
        start_from: str = "middle",
        spacing_mm: Tuple[float, float, float] = None,
        percentile: float = 1,
    ):
        """Render difference maps over CT slices and save to PNGs."""

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        warped_vol = warp_image(moving_vol, dvf, spacing_mm)

        fixed_vol_np = fixed_vol.detach().cpu().numpy()  # [D,H,W]
        warped_vol_np = warped_vol.detach().cpu().numpy()  # [D,H,W]
        moving_vol_np = moving_vol.detach().cpu().numpy()  # [D,H,W]

        saved = []

        for view, axis, idx in self._iter_slices(
            fixed_vol, views, num_slices, start_from
        ):
            fixed_slice = np.squeeze(get_ct_view(fixed_vol_np, view=view, idx=idx))
            warped_slice = np.squeeze(get_ct_view(warped_vol_np, view=view, idx=idx))
            moving_slice = np.squeeze(get_ct_view(moving_vol_np, view=view, idx=idx))

            abs_warped_fixed_difference = np.abs(warped_slice - fixed_slice)
            abs_moving_fixed_difference = np.abs(moving_slice - fixed_slice)

            _, _, extent = self._make_grid(fixed_slice, axis, spacing_mm)

            origin = "lower"
            if view == "axial":
                extent = (extent[0], extent[1], extent[3], extent[2])  # flip top-bottom
                origin = "upper"

            fig, fig_axes = plt.subplots(1, 4, figsize=(15, 4))

            for ax in fig_axes:
                ax.title.set_fontsize(12)

            vmin = np.percentile(fixed_slice, percentile)
            vmax = np.percentile(fixed_slice, 100 - percentile)

            im0 = fig_axes[0].imshow(
                fixed_slice,
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
                extent=extent,
                origin=origin,
            )
            fig_axes[0].set_title("Fixed")
            fig_axes[0].axis("off")
            ssim_warped = ssim(
                fixed_slice,
                warped_slice,
                data_range=warped_slice.max() - warped_slice.min(),
            )
            im1 = fig_axes[1].imshow(
                warped_slice,
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
                extent=extent,
                origin=origin,
            )
            fig_axes[1].set_title(f"Warped\nSSIM: {ssim_warped:.3f}")
            fig_axes[1].axis("off")

            im2 = fig_axes[2].imshow(
                abs_warped_fixed_difference,
                cmap=self.diffmap_cmap,
                vmin=0,
                vmax=0.3,
                extent=extent,
                origin=origin,
            )
            fig_axes[2].set_title("|warped - fixed|")
            fig_axes[2].axis("off")

            im3 = fig_axes[3].imshow(
                abs_moving_fixed_difference,
                cmap=self.diffmap_cmap,
                vmin=0,
                vmax=0.3,
                extent=extent,
                origin=origin,
            )
            fig_axes[3].set_title("|moving - fixed|")
            fig_axes[3].axis("off")

            fname = Path(out_dir) / f"{view}_slice_{idx}_diffmap.png"
            plt.savefig(fname, bbox_inches=None, dpi=200)
            plt.close(fig)
            saved.append(fname)
        return saved
