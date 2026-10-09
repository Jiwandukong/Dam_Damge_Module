"""Group adjacent video frames using geometrically consistent image features."""
from pathlib import Path
import re
import cv2
import numpy as np

GROUPING_PARAMETERS = dict(
    method='SIFT + adjacent-frame RANSAC homography',
    resize_px=[800, 450], max_frame_gap=1500,
    descriptor_ratio=.75, minimum_inliers=8,
    minimum_inlier_fraction=.35, minimum_image_coverage=.03,
)


def frame_number(path):
    match = re.search(r'frame_(\d+)', Path(path).stem)
    return int(match.group(1)) if match else None


def group_similar_frames(paths, seed=42):
    paths = sorted(map(Path, paths), key=lambda p: (frame_number(p) or 0, p.name))
    if not paths or len({p.name for p in paths}) != len(paths):
        raise ValueError('Frame grouping needs unique image filenames')
    cv2.setNumThreads(1)
    sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=.015)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    features = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError('Cannot read frame: '+str(path))
        image = cv2.resize(image, tuple(GROUPING_PARAMETERS['resize_px']))
        image = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(image)
        keypoints, descriptors = sift.detectAndCompute(image, None)
        features.append((keypoints, descriptors))
    groups = [[paths[0].name]]
    comparisons = []
    for i in range(1, len(paths)):
        previous, current = paths[i-1], paths[i]
        a, b = frame_number(previous), frame_number(current)
        gap = b-a if a is not None and b is not None else None
        result = dict(previous=previous.name, current=current.name, frame_gap=gap,
                      matches=0, inliers=0, inlier_fraction=0., image_coverage=0., similar=False)
        ka, da = features[i-1]; kb, db = features[i]
        if (gap is not None and 0 < gap <= GROUPING_PARAMETERS['max_frame_gap']
                and da is not None and db is not None and len(db) >= 2):
            matches = [pair[0] for pair in matcher.knnMatch(da, db, k=2)
                       if len(pair) == 2 and pair[0].distance < GROUPING_PARAMETERS['descriptor_ratio']*pair[1].distance]
            result['matches'] = len(matches)
            if len(matches) >= GROUPING_PARAMETERS['minimum_inliers']:
                source = np.array([ka[m.queryIdx].pt for m in matches])
                destination = np.array([kb[m.trainIdx].pt for m in matches])
                cv2.setRNGSeed(int(seed) & 0x7fffffff)
                homography, mask = cv2.findHomography(source, destination, cv2.RANSAC, 3.,
                                                       maxIters=2000, confidence=.995)
                if homography is not None and mask is not None:
                    inliers = mask.reshape(-1).astype(bool)
                    result['inliers'] = int(inliers.sum())
                    result['inlier_fraction'] = result['inliers']/len(matches)
                    if result['inliers'] >= 4:
                        hull = cv2.convexHull(source[inliers].astype(np.float32))
                        result['image_coverage'] = float(cv2.contourArea(hull)/(800*450))
                    result['similar'] = (
                        result['inliers'] >= GROUPING_PARAMETERS['minimum_inliers']
                        and result['inlier_fraction'] >= GROUPING_PARAMETERS['minimum_inlier_fraction']
                        and result['image_coverage'] >= GROUPING_PARAMETERS['minimum_image_coverage'])
        if result['similar']:
            groups[-1].append(current.name)
        else:
            groups.append([current.name])
        comparisons.append(result)
    return groups, comparisons
