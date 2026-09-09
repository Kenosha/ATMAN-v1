"""Reproduce a small portfolio from ATMAN's saved broadcast-match features."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time

MATCH = "2021-7341-ms001-brest-brandon-nakashima-vs-joao-sousa"


def load_points(data_root, match=MATCH, store="driver_run_store", sample=False):
    import pandas as pd
    from ATMANpy.types.Abstract import Points, read_in_data_tables

    if sample:
        sample_path = os.path.join(data_root, "examples", "brest_sample.csv")
        table = pd.read_csv(sample_path).set_index(["match_idx", "frame_num"])
    else:
        table = read_in_data_tables([{"video_id": match, "store_name": store}])
    return table, Points.from_datatable(table)


def summarize(table, points):
    import numpy as np

    shots = points.flatten()
    serves = shots.filter("is_serve")
    first = shots.filter("is_first_serve")
    speeds = shots.get_initial_speed()
    spins = shots.get_initial_spin()
    bounces = shots.get_first_bounce_position()
    rows = []
    for shot in shots:
        bounce = shot.get_first_bounce_position()
        rows.append({
            "point": shot.get_point_index(), "shot": shot.get_shot_index(),
            "hitter": shot.get_hit_player_name(), "event": shot.get_hit_event(),
            "is_serve": bool(shot.is_serve()), "is_first_serve": bool(shot.is_first_serve()),
            "start_frame": shot.get_hit_frame(), "stop_frame": shot.get_next_hit_frame(),
            "speed_kmh": float(shot.get_initial_speed()),
            "spin_rpm": float(shot.get_initial_spin()),
            "launch_angle_deg": float(shot.get_launch_angle()),
            "bounce_x_court_units": float(bounce[0]) if bounce[0] is not None else float("nan"),
            "bounce_y_court_units": float(bounce[1]) if bounce[1] is not None else float("nan"),
        })
    # This stable numeric fingerprint is also produced against the original checkout.
    values = np.column_stack([speeds, spins, bounces]).astype("<f8")
    values = np.nan_to_num(values, nan=-999999.0)
    fingerprint = hashlib.sha256(np.round(values, 8).tobytes()).hexdigest()
    summary = {
        "feature_rows": len(table), "points": len(points), "shots": len(shots),
        "serves": len(serves), "first_serves": len(first),
        "returns": len(points.get_returns()),
        "shots_over_150_kmh": len(shots.filter("get_initial_speed", is_above=150)),
        "players": sorted(set(row["hitter"] for row in rows)),
        "mean_speed_kmh": float(np.nanmean(speeds)),
        "mean_spin_rpm": float(np.nanmean(spins)),
        "numeric_sha256": fingerprint,
    }
    return summary, rows


def save_image(path, frame):
    import cv2
    if not cv2.imwrite(path, frame.array):
        raise RuntimeError(f"Could not write image: {path}")


def write_video(path, frames, fps=25):
    """Encode the frames rendered by ATMAN, without buffering the whole video."""
    import numpy as np

    iterator = iter(frames)
    first = next(iterator)
    height, width = first.array.shape[:2]
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{width}x{height}",
        "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-threads", "1", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", path,
    ]
    count = 0
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in _prepend(first, iterator):
            array = np.ascontiguousarray(frame.array, dtype=np.uint8)
            if array.shape != (height, width, 3):
                raise ValueError("Video frames must have the same dimensions.")
            process.stdin.write(array.tobytes())
            count += 1
        process.stdin.close()
        error = process.stderr.read().decode("utf-8", errors="replace")
        if process.wait():
            raise RuntimeError(f"ffmpeg failed: {error}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stderr.close()
    return count


def _prepend(first, rest):
    yield first
    # The historical FramesIterator implements __next__ but not __iter__.
    while True:
        try:
            yield next(rest)
        except StopIteration:
            return


def render_portfolio(points, output, video=None, max_frames=150,
                     broadcast_camera=None, detections=None):
    import cv2
    import numpy as np
    from ATMANpy.types.Overlays import Graph, ScatterChart
    from ATMANpy.types.Visual3D import Camera, View, Scene
    import ATMANpy.auxil.constants as const
    import ATMANpy.funcs.vis_funcs as vis

    shots = points.flatten()
    serves = shots.filter("is_serve").enforce_direction("Top")
    camera = Camera.calibrated_brest_behind()
    court = Scene.make_court(singles_net=False, net_mesh_density=3, net_alpha=0.35)
    court_view = View.make(camera, court, resolution=0.8)

    scatter = vis.scatter_markmap(serves, one_color=const.BGR["ball_yellow"], mark_size=3)
    scatter_frame = court_view.combine(View.make(camera, scatter, resolution=0.8)).snap()
    scatter_frame = scatter_frame.put("Serve bounce positions | normalized direction", size=0.55)
    save_image(os.path.join(output, "serve_placement.png"), scatter_frame)

    density = vis.kde_heatmap(serves, resolution=0.35, bandwidth=35,
                              one_color=const.BGR["bright_blue"])
    density_frame = View.make(camera, density).combine(court_view).snap()
    density_frame = density_frame.put("Serve bounce density | historical estimates", size=0.55)
    save_image(os.path.join(output, "serve_density.png"), density_frame)

    colors = [const.BGR["ball_yellow"], const.BGR["bright_blue"]]
    players = sorted(set(shots.get_hit_player_name()))
    charts = []
    for player, color in zip(players, colors):
        selected = shots.filter(lambda shot: shot.get_hit_player_name() == player)
        xs, ys = selected.get_initial_speed(), selected.get_initial_spin()
        finite = np.isfinite(xs) & np.isfinite(ys)
        charts.append(ScatterChart(xs[finite], ys[finite], color=color, thickness=0.08))
    graph = Graph(charts, dimensions=(960, 540), resolution=20,
                  color=const.BGR["bright_grey"],
                  header_text="Shot speed and spin (estimated)",
                  x_label="Speed (km/h)", y_label="Spin (RPM)",
                  label_size=1.1, tick_label_size=0.9)
    graph_frame = graph.render()
    for idx, (player, color) in enumerate(zip(players, colors)):
        graph_frame = graph_frame.put(player, where=(620, 68 + 23 * idx),
                                       color=color, size=0.55)
    save_image(os.path.join(output, "speed_spin.png"), graph_frame)

    point = next(point for point in points if len(point) >= 2)
    scenes = []
    frame_numbers = []
    for shot in point:
        splines = shot.feat_spline_cumul(which_endpoint="penultimate", thickness=4,
                                        color=const.BGR["ball_yellow"])
        for number, lines in zip(splines.get_frames(), splines.values()):
            valid = [line for line in lines if line is not None
                     and np.isfinite(line.leftPt.coords).all()
                     and np.isfinite(line.rightPt.coords).all()]
            scenes.append(Scene(valid))
            frame_numbers.append(int(number))
            if len(scenes) >= max_frames:
                break
        if len(scenes) >= max_frames:
            break
    if not scenes:
        raise ValueError("No finite trajectory frames available for the example point.")

    # Exercise the same animation utility used for the original portfolio videos.
    animation, _ = vis.animate_sequence(scenes, start_cam=camera, stop_cam=camera,
                                        background_scene=court, background_resolution=0.8,
                                        argument_resolution=0.8)
    virtual_path = os.path.join(output, "trajectory.mp4")
    count = write_video(virtual_path, animation)
    preview_index = max(range(count), key=lambda idx: len(scenes[idx]))
    save_image(os.path.join(output, "trajectory.png"), animation[preview_index])
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", virtual_path,
        "-filter_complex_threads", "1", "-filter_complex",
        "fps=10,scale=480:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse",
        "-threads", "1", "-loop", "0", os.path.join(output, "trajectory.gif")
    ], check=True)
    result = {"point_index": point.get_point_index(), "frames": count,
              "first_source_frame": frame_numbers[0], "last_source_frame": frame_numbers[-1],
              "fps": 25, "broadcast_overlay": False}
    if video:
        if broadcast_camera is None:
            raise ValueError("A broadcast overlay needs the matching feature store's camera calibration.")
        from ATMANpy.types.Frames import BGRFrame
        capture = cv2.VideoCapture(video)
        if not capture.isOpened():
            raise ValueError(f"Cannot open local video: {video}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isclose(fps, 25, atol=0.05):
            capture.release()
            raise ValueError("The stored Brest features require the original 25 fps video.")
        def overlays():
            detection_frames = 0
            try:
                for number, scene in zip(frame_numbers, scenes):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, number)
                    ok, array = capture.read()
                    if not ok:
                        raise ValueError(f"Cannot decode source frame {number}.")
                    if (array.shape[1], array.shape[0]) != broadcast_camera.sensor_dims:
                        raise ValueError("The Brest features require matching 640x360 source frames.")
                    frame = View.make(broadcast_camera, scene, resolution=0.8).snap(over_array=array.copy())
                    detection = None if detections is None else detections.maybe_get(number)
                    if detection is not None:
                        frame = frame.put(detection, color=const.BGR["bright_blue"], radius=5, thickness=2)
                        detection_frames += 1
                        if detection_frames == 1:
                            snapshot = BGRFrame(array).put(detection, color=const.BGR["bright_blue"], radius=5, thickness=2)
                            snapshot = snapshot.put("Stored ball detection", where=(10, 22), size=0.6)
                            save_image(os.path.join(output, "stored_detection.png"), snapshot)
                    yield frame
            finally:
                capture.release()
                result["stored_detection_frames"] = detection_frames
        result["overlay_frames"] = write_video(os.path.join(output, "broadcast_overlay.mp4"), overlays())
        result["broadcast_overlay"] = True
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="Data directory; sample mode defaults to the current checkout.")
    parser.add_argument("--match", default=MATCH)
    parser.add_argument("--store", default="driver_run_store")
    parser.add_argument("--sample", action="store_true", help="Use the small included CSV excerpt.")
    parser.add_argument("--video", help="Optional matching local broadcast video for an overlay.")
    parser.add_argument("--output", default="testing_outputs")
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument("--metrics-only", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.max_frames <= 500:
        parser.error("--max-frames must be between 1 and 500")
    if args.sample and args.video:
        parser.error("--video requires the full feature store, including its camera calibration")
    default_root = os.getcwd() if args.sample else os.environ.get("ATMAN_HOME", os.getcwd())
    data_root = os.path.abspath(args.data_root or default_root)
    os.environ["ATMAN_HOME"] = data_root
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    output = os.path.abspath(args.output)
    os.makedirs(output, exist_ok=True)
    import cv2
    cv2.setNumThreads(1)
    started = time.monotonic()
    table, points = load_points(data_root, args.match, args.store, args.sample)
    summary, rows = summarize(table, points)
    for name, selected in [("shots", rows),
                           ("first_serves", [row for row in rows if row["is_first_serve"]]),
                           ("fast_shots", [row for row in rows if row["speed_kmh"] > 150])]:
        with open(os.path.join(output, name + ".csv"), "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(selected)
    broadcast_camera, detections = None, None
    if args.video and not args.metrics_only:
        from ATMANpy.types.Features import ConstFeature, StreamFeature
        from ATMANpy.types.Visual3D import Camera
        matrices = [ConstFeature.from_store(args.match, args.store, featname=name).feature
                    for name in ["M_int", "M_ext"]]
        broadcast_camera = Camera(*matrices, (640, 360))
        detections = StreamFeature.from_store(args.match, args.store, featname="ball_choicePt")
    rendering = None if args.metrics_only else render_portfolio(
        points, output, args.video, max_frames=args.max_frames,
        broadcast_camera=broadcast_camera, detections=detections)
    report = {"match": args.match, "source": "sample excerpt" if args.sample else args.store,
              "metrics": summary, "rendering": rendering,
              "elapsed_seconds": round(time.monotonic() - started, 3),
              "note": "Historical model estimates, not independently verified ground truth."}
    with open(os.path.join(output, "report.json"), "w") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")
    with open(os.path.join(output, "README.md"), "w") as file:
        file.write("# ATMAN v1 demonstration outputs\n\n")
        file.write(f"Source: {args.match}. These are historical model estimates.\n\n")
        file.write(f"{summary['points']} reconstructed points; {summary['shots']} shots; "
                   f"{summary['serves']} serves; {summary['returns']} returns.\n\n")
        file.write("`shots.csv` contains filtered-query inputs and estimated speed, spin and bounce positions. "
                   "`first_serves.csv` and `fast_shots.csv` are filtered subsets. "
                   "`report.json` records counts and the numeric fingerprint used for preservation checks.\n\n")
        if rendering:
            file.write("The PNG charts, court maps and trajectory frames use ATMAN's original visualization classes. "
                       "`trajectory.mp4` and `trajectory.gif` show a reconstructed rally on the virtual court.\n")
            if args.video:
                file.write("`broadcast_overlay.mp4` applies the reconstructed path (yellow) and stored ball detections "
                           "(blue) to the matching local footage using its saved camera calibration. "
                           "`stored_detection.png` shows an original intermediate detection.\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
