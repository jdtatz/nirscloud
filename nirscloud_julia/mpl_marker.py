import matplotlib as mpl
import matplotlib.markers

def centered_text_as_path(s: str) -> mpl.path.Path:
    # fp = mpl.font_manager.FontProperties(size=1, family='monospace')
    fp = mpl.font_manager.FontProperties(size=1)
    # dx, _, _ = mpl.textpath.text_to_path.get_text_width_height_descent("q", fp, ismath=False)
    # dy = font.get_size()
    w, h, _d = mpl.textpath.text_to_path.get_text_width_height_descent(s, fp, ismath=False)
    trans = mpl.transforms.Affine2D.identity().scale(fp.get_size() / mpl.textpath.text_to_path.FONT_SCALE).translate(-w/2, -h/2)
    verts, codes = mpl.textpath.text_to_path.get_text_path(fp, s)
    return mpl.path.Path(verts, codes, closed=False).transformed(trans)


def marker_with_text(base_marker, text: str) -> mpl.markers.MarkerStyle:
    if not isinstance(base_marker, mpl.markers.MarkerStyle):
        assert hasattr(base_marker, '__hash__') and base_marker in mpl.markers.MarkerStyle.markers
        base_marker = mpl.markers.MarkerStyle(base_marker)
    # ignoring alternate path for now
    base_path = base_marker.get_path().transformed(base_marker.get_transform())
    text_path = centered_text_as_path(text).transformed(mpl.transforms.Affine2D.identity().scale(3/4))
    marker_path = mpl.path.Path.make_compound_path(base_path, text_path)
    return mpl.markers.MarkerStyle(marker_path)
