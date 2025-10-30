import cv2 as cv
import numpy as np

def click_labeled_points(myFrm, labels, window_name="Select points"):
    """
    Click points in the order described by `labels`, one label per point.

    Args:
        myFrm: frame object with .to_BGR().array.copy()
        labels: list[str], e.g. ["L post base", "L post top", "Net mid-top", "R post base", "R post top"]
        window_name: window title

    Controls:
        - Left-click: add the next point (up to len(labels))
        - Right-click: undo last point
        - 'c': clear all points
        - Enter/Space: confirm (only when all points are selected)
        - Esc or 'q': cancel (returns None)

    Returns:
        np.ndarray of shape (N, 2), dtype float32, with (x,y) in click order
        or None if cancelled.
    """
    if not labels or not isinstance(labels, (list, tuple)):
        raise ValueError("`labels` must be a non-empty list/tuple of strings.")
    
    try:
        showFrm = myFrm.to_BGR()
    except AttributeError:
        showFrm = myFrm

    frmArr = showFrm.array.copy()  # BGR ndarray
    points = []

    def draw_overlay(img, pts):
        vis = img.copy()
        for i, (x, y) in enumerate(pts):
            # Large, high-contrast marker + index
            cv.circle(vis, (x, y), 12, (0, 0, 0), 3)       # outer ring
            cv.circle(vis, (x, y), 10, (0, 255, 255), -1)  # filled center (yellow)
            cv.putText(vis, str(i + 1), (x + 14, y - 14),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv.LINE_AA)
        return vis

    def on_mouse(event, x, y, flags, userdata):
        if event == cv.EVENT_LBUTTONDOWN:
            if len(points) < len(labels):
                points.append((x, y))
        elif event == cv.EVENT_RBUTTONDOWN:
            if points:
                points.pop()

    cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)  # real-size display
    cv.setMouseCallback(window_name, on_mouse)

    while True:
        disp = draw_overlay(frmArr, points)

        if len(points) < len(labels):
            prompt = f"{labels[len(points)]} ({len(points)+1}/{len(labels)})"
        else:
            prompt = "All points selected. Press Enter/Space to confirm, or Right-click to adjust."

        # Top prompt
        cv.putText(disp, prompt, (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)

        # Bottom help
        help_text = "Left: add | Right: undo | c: clear | Enter/Space: confirm | Esc/q: cancel"
        cv.putText(disp, help_text, (10, disp.shape[0] - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv.LINE_AA)

        cv.imshow(window_name, disp)
        key = cv.waitKey(20) & 0xFFFF

        if key in (27, ord('q')):          # Esc or 'q' → cancel
            cv.destroyWindow(window_name)
            return None
        elif key in (ord('c'), ord('C')):  # clear
            points.clear()
        elif key in (13, 32):              # Enter or Space → confirm if done
            if len(points) == len(labels):
                cv.destroyWindow(window_name)
                return np.array(points, dtype=np.float32)
