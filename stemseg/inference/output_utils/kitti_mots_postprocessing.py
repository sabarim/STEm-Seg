from argparse import ArgumentParser
from collections import defaultdict

from glob import glob
import pycocotools.mask as masktools
import os


CAR_CLASS_ID = 1
PERSON_CLASS_ID = 2

# default parameter values
DEFAULT_MIN_AREA_CAR = 150
DEFAULT_MIN_AREA_PEDESTRIAN = 250

DEFAULT_MIN_TRACK_LENGTH_CAR = 3
DEFAULT_MIN_TRACK_LENGTH_PEDESTRIAN = 10

DEFAULT_MIN_AREA_RATIO_CAR = 0.35
DEFAULT_MIN_AREA_RATIO_PEDESTRIAN = 0.2

DEFAULT_MAX_TIME_BREAK_RATIO_CAR = 0.3
DEFAULT_MAX_TIME_BREAK_RATIO_PEDESTRIAN = 0.5


class Detection(object):
    def __init__(self, frame_id, track_id, class_id, mask):
        self.frame_id = frame_id
        self.track_id = track_id
        self.class_id = class_id
        self._mask = mask

    def as_txt(self):
        return "{} {} {} {} {} {}".format(
            self.frame_id, self.track_id, self.class_id, self._mask['size'][0], self._mask['size'][1],
            self._mask['counts'].decode("utf-8")
        )

    @property
    def mask(self):
        return masktools.decode(self._mask)

    @property
    def pixel_area(self):
        return masktools.area(self._mask)

    @property
    def bbox_area(self):
        x, y, w, h = masktools.toBbox(self._mask)
        return w*h

    @property
    def pixel_bbox_area_ratio(self):
        bbox_area = self.bbox_area
        if bbox_area == 0:
            return 0.
        else:
            return float(self.pixel_area) / float(self.bbox_area)

    @classmethod
    def from_txt(cls, txt):
        fields = txt.strip().split(' ')
        return cls(
            int(fields[0]), int(fields[1]), int(fields[2]),
            {
                "size": (int(fields[3]), int(fields[4])),
                "counts": fields[5].encode("utf-8")
            }
        )


def detections_to_tracks(detections):
    tracks = defaultdict(list)
    for det in detections:
        tracks[det.track_id].append(det)

    for track_id in tracks:
        tracks[track_id] = sorted(tracks[track_id], key=lambda d: d.frame_id)

    return list(tracks.values())


def compute_track_span(track):
    min_t = min(track, key=lambda det: det.frame_id).frame_id
    max_t = max(track, key=lambda det: det.frame_id).frame_id
    return max_t - min_t + 1


def compute_nbr_time_breaks(track):
    n_breaks = 0

    for i in range(len(track)-1):
        n_breaks += int(track[i+1].frame_id - track[i].frame_id > 1)

    return n_breaks


def filter_tracks_by_length(detections, min_track_length_car, min_track_length_person):
    tracks = detections_to_tracks(detections)

    filtered_dets = []
    for t in tracks:
        if t[0].class_id == CAR_CLASS_ID and len(t) < min_track_length_car:
            continue
        elif t[0].class_id == PERSON_CLASS_ID and len(t) < min_track_length_person:
            continue

        filtered_dets.extend(t)

    return filtered_dets


def filter_tracks_by_time_breaks(detections, max_time_break_ratio_car, max_time_break_ratio_person):
    tracks = detections_to_tracks(detections)

    filtered_dets = []
    for t in tracks:
        # print(float(compute_nbr_time_breaks(t)) / float(len(t)))
        if t[0].class_id == CAR_CLASS_ID and (float(compute_nbr_time_breaks(t)) / float(len(t))) > max_time_break_ratio_car:
            continue
        elif t[0].class_id == PERSON_CLASS_ID and (float(compute_nbr_time_breaks(t)) / float(len(t))) > max_time_break_ratio_person:
            continue

        filtered_dets.extend(t)

    return filtered_dets


def filter_detections_by_area(detections, min_car_area, min_person_area):
    return [
        det for det in detections
        if (det.class_id == CAR_CLASS_ID and det.pixel_area >= min_car_area) or
           (det.class_id == PERSON_CLASS_ID and det.pixel_area >= min_person_area)
    ]


def filter_detections_by_area_ratio(detections, min_ratio_cars, min_ratio_persons):
    return [
        det for det in detections
        if (det.class_id == CAR_CLASS_ID and det.pixel_bbox_area_ratio > min_ratio_cars) or
           (det.class_id == PERSON_CLASS_ID and det.pixel_bbox_area_ratio > min_ratio_persons)
    ]


def main(**kwargs):
    result_files = sorted(glob(os.path.join(kwargs['results_dir'], "????.txt")))
    output_dir = kwargs['results_dir'] + "_{}".format(kwargs.get('output_dir_suffix', 'nms'))
    os.makedirs(output_dir, exist_ok=True)

    min_area_car = kwargs.get('min_car_area', DEFAULT_MIN_AREA_CAR)
    min_area_pedestrian = kwargs.get('min_person_area', DEFAULT_MIN_AREA_PEDESTRIAN)

    min_area_ratio_car = kwargs.get('min_area_ratio_car', DEFAULT_MIN_AREA_RATIO_CAR)
    min_area_ratio_pedestrian = kwargs.get('min_area_ratio_person', DEFAULT_MIN_AREA_RATIO_PEDESTRIAN)

    min_track_length_car = kwargs.get('min_track_length_car', DEFAULT_MIN_TRACK_LENGTH_CAR)
    min_track_length_pedestrian = kwargs.get('min_track_length_person', DEFAULT_MIN_TRACK_LENGTH_PEDESTRIAN)

    max_time_break_ratio_car = kwargs.get('max_time_break_ratio_car', DEFAULT_MAX_TIME_BREAK_RATIO_CAR)
    max_time_break_ratio_pedestrian = kwargs.get('max_time_break_ratio_person', DEFAULT_MAX_TIME_BREAK_RATIO_PEDESTRIAN)

    for f in result_files:
        seq_file_name = os.path.split(f)[-1]

        print("Processing {}".format(seq_file_name))
        with open(f, 'r') as fh:
            detections = [Detection.from_txt(det_txt) for det_txt in fh.readlines()]

        detections = filter_detections_by_area(detections, min_area_car, min_area_pedestrian)

        detections = filter_detections_by_area_ratio(detections, min_area_ratio_car, min_area_ratio_pedestrian)

        detections = filter_tracks_by_time_breaks(detections, max_time_break_ratio_car, max_time_break_ratio_pedestrian)

        detections = filter_tracks_by_length(detections, min_track_length_car, min_track_length_pedestrian)

        with open(os.path.join(output_dir, seq_file_name), 'w') as fh:
            fh.writelines([det.as_txt() + "\n" for det in detections])

    print("Results after applying NMS written to: {}".format(output_dir))


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('results_dir')
    parser.add_argument('--min_car_area', '-mca',                    type=int, default=DEFAULT_MIN_AREA_CAR)
    parser.add_argument('--min_person_area', '-mpa',                 type=int, default=DEFAULT_MIN_AREA_PEDESTRIAN)

    parser.add_argument('--min_track_length_car', '-mtlc',           type=int, default=DEFAULT_MIN_TRACK_LENGTH_CAR)
    parser.add_argument('--min_track_length_person', '-mtlp',        type=int, default=DEFAULT_MIN_TRACK_LENGTH_PEDESTRIAN)

    parser.add_argument('--min_area_ratio_car', '-marc',             type=float, default=DEFAULT_MIN_AREA_RATIO_CAR)
    parser.add_argument('--min_area_ratio_person', '-marp',          type=float, default=DEFAULT_MIN_AREA_RATIO_PEDESTRIAN)

    parser.add_argument('--max_time_break_ratio_car', '-mtbrc',      type=float, default=DEFAULT_MAX_TIME_BREAK_RATIO_CAR)
    parser.add_argument('--max_time_break_ratio_person', '-mtbrp',   type=float, default=DEFAULT_MAX_TIME_BREAK_RATIO_PEDESTRIAN)

    main(**vars(parser.parse_args()))
