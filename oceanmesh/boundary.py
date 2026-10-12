import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shapely

from .edges import get_winded_boundary_edges

__all__ = [
    "gdf_order_boundary_sections",
    "gdf_simple_assign_exterior_ibtype",
    "gdf_simple_assign_ibtype_by_thresh",
    "identify_ocean_boundary_sections",
    "mesh_union_exterior_feat",
    "mesh_union_interior_feat",
    "mesh_union_polygon_feat",
    "naive_exterior_and_island_boundary_classification",
    "ordered_exterior_gdf",
    "ordered_exterior_point_feat",
    "plot_gdf_ibtype",
]


def mesh_union_polygon_feat(points: np.ndarray, cells: np.ndarray, crs=None):
    """[TODO:description]

    Args:
        crs ([TODO:parameter]): [TODO:description]
        points: [TODO:description]
        cells: [TODO:description]

    Returns:
        [TODO:return]
    """
    elats = points[cells.ravel(), 1]
    elons = points[cells.ravel(), 0]
    ncell = cells.shape[0]
    _elons = elons.reshape((ncell, 3))
    _elats = elats.reshape((ncell, 3))
    _pinput = [shapely.Polygon(tuple(zip(_[0], _[1]))) for _ in zip(_elons, _elats)]
    tris_gdf = gpd.GeoDataFrame(geometry=_pinput, crs=crs)
    assert tris_gdf.is_valid.all()
    return tris_gdf.union_all()


def mesh_union_exterior_feat(points: np.ndarray, cells: np.ndarray, crs=None):
    """[TODO:description]

    Args:
        crs ([TODO:parameter]): [TODO:description]
        points: [TODO:description]
        cells: [TODO:description]

    Returns:
        [TODO:return]
    """
    domain_poly = mesh_union_polygon_feat(points, cells, crs=crs)
    return (domain_poly.exterior, domain_poly)


def ordered_exterior_point_feat(points: np.ndarray, cells: np.ndarray, crs=None):
    """[TODO:description]

    Args:
        crs ([TODO:parameter]): [TODO:description]
        points: [TODO:description]
        cells: [TODO:description]

    Returns:
        [TODO:return]
    """
    ext_line, poly = mesh_union_exterior_feat(points, cells, crs=None)
    return (ext_line.xy, ext_line, poly)


def ordered_exterior_gdf(gdf, ext_feat):
    """[TODO:description]

    Args:
        gdf ([TODO:parameter]): [TODO:description]
        ext_feat ([TODO:parameter]): [TODO:description]

    Returns:
        [TODO:return]
    """
    return gdf.loc[ext_feat, :]


def gdf_simple_assign_exterior_ibtype(gdf: gpd.GeoDataFrame, ext_feat, ibtype=20):
    """[TODO:description]

    Args:
        ext_feat ([TODO:parameter]): [TODO:description]
        ibtype ([TODO:parameter]): [TODO:description]
        gdf: [TODO:description]

    Returns:
        [TODO:return]
    """
    which_ext_nodes = gdf.geometry.intersects(ext_feat).values
    gdf.loc[which_ext_nodes, "ibtype"] = ibtype
    return gdf


def gdf_simple_assign_ibtype_by_thresh(
    gdf: gpd.GeoDataFrame, val, column="depth", ibtype=-1, assign_lessthan=True
):
    """[TODO:description]

    Args:
        val ([TODO:parameter]): [TODO:description]
        column ([TODO:parameter]): [TODO:description]
        ibtype ([TODO:parameter]): [TODO:description]
        assign_lessthan ([TODO:parameter]): [TODO:description]
        gdf: [TODO:description]

    Returns:
        [TODO:return]
    """
    if assign_lessthan:
        which_nodes = gdf[column] < val
    else:
        which_nodes = gdf[column] > val
    gdf.loc[which_nodes, "ibtype"] = -1
    return gdf


def mesh_union_interior_feat(domain_poly):
    """[TODO:description]

    Args:
        domain_poly ([TODO:parameter]): [TODO:description]

    Returns:
        [TODO:return]
    """
    _interior_nodes = shapely.unary_union(domain_poly.interiors)
    return _interior_nodes


def gdf_order_boundary_sections(
    gdf: gpd.GeoDataFrame,
    ext_feat,
    other_bounds: list[tuple[int, shapely.LineString]] = [],
):
    """[TODO:description]

    Args:
        ext_feat ([TODO:parameter]): [TODO:description]
        gdf: [TODO:description]
        other_bounds: [TODO:description]

    Returns:
        [TODO:return]
    """
    bound_section_counts = {}
    # Exterior tidal and land boundaries
    # uses existing ibtype column data
    edf = gdf.loc[ext_feat]
    vals = edf.ibtype.values
    node = edf.node.values
    geoindex = edf.index.values
    splits = np.where(np.abs((vals[1:] - vals[:-1])) > 0)[0] + 1
    splitibt = np.split(vals, splits)
    splitnode = np.split(node, splits)
    splitidx = np.split(geoindex, splits)
    for _ib, _nodes, _idx in zip(splitibt, splitnode, splitidx):
        _ibtype = int(_ib[0])
        if _ibtype not in bound_section_counts:
            bound_section_counts[_ibtype] = 1
        else:
            bound_section_counts[_ibtype] += 1
        gdf.loc[_idx, "ibtype"] = _ibtype
        gdf.loc[_idx, "ib_section"] = bound_section_counts[_ibtype]
        gdf.loc[_idx, "ib_order"] = np.arange(_nodes.shape[0])

    # Internal island and other boundaries not on exterior ring
    for _thisbound in other_bounds:
        _ibtype, _thismultigeom = _thisbound
        for _ in _thismultigeom.geoms:
            geom_ordered_intersection = gpd.points_from_xy(x=_.xy[0], y=_.xy[1])
            if _ibtype not in bound_section_counts:
                bound_section_counts[_ibtype] = 1
            else:
                bound_section_counts[_ibtype] += 1
            gdf.loc[geom_ordered_intersection, "ibtype"] = _ibtype
            gdf.loc[geom_ordered_intersection, "ib_section"] = bound_section_counts[
                _ibtype
            ]
            gdf.loc[geom_ordered_intersection, "ib_order"] = np.arange(
                gdf.loc[geom_ordered_intersection].shape[0]
            )
    return gdf


def plot_gdf_ibtype(
    points: np.ndarray,
    cells: np.ndarray,
    gdf: gpd.GeoDataFrame,
    xlim=None,
    ylim=None,
    mesh_style={"color": "k", "lw": 0.04},
    ib_styles={
        -1: {"c": "r", "s": "5."},
        20: {"c": "navy", "s": 0.4},
        21: {"c": "g", "s": 3.0},
    },
):
    """[TODO:description]

    Args:
        xlim ([TODO:parameter]): [TODO:description]
        ylim ([TODO:parameter]): [TODO:description]
        mesh_style ([TODO:parameter]): [TODO:description]
        ib_styles ([TODO:parameter]): [TODO:description]
        points: [TODO:description]
        cells: [TODO:description]
        gdf: [TODO:description]

    Returns:
        [TODO:return]
    """
    plt.triplot(points[:, 0], points[:, 1], cells, **mesh_style)
    for z, ibtype in enumerate(ib_styles.keys()):
        _ibgdf = gdf.loc[gdf.ibtype == ibtype]
        plt.scatter(
            _ibgdf.lon.values, _ibgdf.lat.values, **ib_styles[ibtype], zorder=10 + z
        )
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    return plt.gca()


def naive_exterior_and_island_boundary_classification(
    points: np.ndarray,
    cells: np.ndarray,
    crs=None,
    ibtype={"interior": 21, "mainland": 20, "tide_elev": -1},
    tide_elev_bound_depth_thresh=-50.0,
    depth_column: str = "depth",
):
    """[TODO:description]

    Args:
        crs ([TODO:parameter]): [TODO:description]
        ibtype ([TODO:parameter]): [TODO:description]
        tide_elev_bound_depth_thresh ([TODO:parameter]): [TODO:description]
        points: [TODO:description]
        cells: [TODO:description]
        depth_column: [TODO:description]

    Returns:
        [TODO:return]
    """
    ext_point, ext_line, domain_poly = ordered_exterior_point_feat(
        points, cells, crs=crs
    )
    gdf = _node2gdf(points, cells, crs=crs)
    gdf = gdf_simple_assign_exterior_ibtype(gdf, ext_point, ibtype=ibtype["exterior"])
    gdf.loc[ext_point, :] = gdf_simple_assign_ibtype_by_thresh(
        gdf.loc[ext_point, :],
        tide_elev_bound_depth_thresh,
        column=depth_column,
        ibtype=ibtype["tide_elev"],
    )
    _interior = mesh_union_interior_feat(domain_poly)
    gdf = gdf_order_boundary_sections(
        gdf, ext_point, other_bounds=[(ibtype["interior"], _interior)]
    )
    return gdf


def _node2gdf(points: np.ndarray, cells: np.ndarray, crs=None):
    """[TODO:description]

    Args:
        crs ([TODO:parameter]): [TODO:description]
        points: [TODO:description]
        cells: [TODO:description]

    Returns:
        [TODO:return]
    """
    gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(x=points[:, 0], y=points[:, 1]), crs=crs
    )
    gdf["depth"] = np.nan
    gdf["lat"] = points[:, 1]
    gdf["lon"] = points[:, 0]
    gdf["node"] = np.arange(points.shape[0])
    gdf["ibtype"] = -99
    gdf["bound_sort"] = np.nan
    gdf.index = gdf.geometry
    return gdf


def identify_ocean_boundary_sections(
    points,
    cells,
    topobathymetry,
    depth_threshold=-50.0,
    min_nodes_threshold=10,
    plot=False,
):
    """Identify the contiguous sections on the ocean boundary based on depth
    that could be forced in a numerical model as ocean-type boundaries (e.g., elevation-specified)

    Parameters
    ----------
    points: numpy.ndarray
        Array of points (x,y)
    cells : numpy.ndarray
        Array of cells
    topobathymetry : numpy.ndarray
        Array of topobathymetry values (depth below datum negative)
    depth_threshold : float, optional
        Depth threshold to identify ocean boundary nodes, by default -50 m below the datum
    min_nodes_threshold : int, optional
        Minimum number of nodes to be considered a boundary section, by default 10
    plot : bool, optional
        Plot the mesh and the identified boundary sections, by default False

    Returns
    --------
    boundary_sections : list
        List of tuples of the nodes that define the ocean boundary sections
        Note these map back into the points array.

    """
    # Identify the nodes on the boundary of the mesh
    boundary_edges = get_winded_boundary_edges(cells)
    boundary_edges = boundary_edges.flatten()
    unique_indexes = np.unique(boundary_edges, return_index=True)[1]
    boundary_nodes_unmasked = [
        boundary_edges[unique_index] for unique_index in sorted(unique_indexes)
    ]
    # Define a boolean array of valid nodes
    bathymetry_on_boundary = topobathymetry[boundary_nodes_unmasked]
    # Append a NaN value to the array to align with the original
    bathymetry_on_boundary = np.append(bathymetry_on_boundary, np.nan)
    stops = np.nonzero(bathymetry_on_boundary <= depth_threshold)[0]

    # Plot the mesh
    if plot:
        fig, ax = plt.subplots()
        ax.triplot(points[:, 0], points[:, 1], cells, color="k", lw=0.1)

    first = True
    boundary_sections = []
    start_node = None
    end_node = None
    for idx, (s1, s2) in enumerate(zip(stops[:-1], stops[1:])):
        if s2 - s1 < min_nodes_threshold:
            if first:
                start_node = s1
                first = False
            # We've reached the end of the list
            elif idx == len(stops) - 2:
                # Append the start and end nodes to the boundary sections list
                end_node = s2
                boundary_sections.append([start_node, end_node])
            # Its not the end and we haven't found a section yet
            else:
                end_node = s2
        elif s2 - s1 >= min_nodes_threshold and not first:
            # Append the start and end nodes to the boundary sections list
            boundary_sections.append([start_node, end_node])
            # Reset the start node, the last node didn't satisfy the threshold
            # and it appears we have a new section
            start_node = s1
            first = True
        # We've reached the end of the list
        elif idx == len(stops) - 2:
            # Save the end node
            end_node = s2
            # Append the start and end nodes to the boundary sections list and finish
            boundary_sections.append([start_node, end_node])
    if plot:
        for s1, s2 in boundary_sections:
            ax.scatter(
                points[boundary_nodes_unmasked[s1:s2], 0],
                points[boundary_nodes_unmasked[s1:s2], 1],
                5,
                c="r",
            )
            ax.set_title("Identified ocean boundary sections")
        plt.show()

    # Map back to the original node indices associated with the points array
    boundary_sections = [
        (boundary_nodes_unmasked[s1], boundary_nodes_unmasked[s2])
        for s1, s2 in boundary_sections
    ]
    return boundary_sections
