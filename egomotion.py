import cv2
import time
import math
from feature_transform import feature_transform


MIN_MATCH_COUNT = 10

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()


def match(img1, img2):
    try:
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.

        res = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                k1 = kp1[m.queryIdx]
                k2 = kp2[m.trainIdx]
                pt1 = tuple(int(x) for x in k1.pt)
                pt2 = tuple(int(x) for x in k2.pt)
                res.append((pt1, pt2))

        return res
    except:
        return []


def filter_outliers(matches, thresh=100):
    res = []

    for pt1, pt2 in matches:
        dx = pt1[0] - pt2[0]
        dy = pt1[1] - pt2[1]
        dist2 = dx * dx + dy * dy
        if dist2 < thresh * thresh:
            res.append((pt1, pt2))
    return res


def get_transoform(matches):
    l, r = ([(0, 0)], [(0, 0)])
    if len(matches) > 0:
        l, r = zip(*matches)
    return feature_transform(l, r)


def visualize_matching(img1, matches):
    for pt1, pt2 in matches:
        cv2.line(img1, pt1, pt2, (0, 255, 0))
    return img1


def test():
    cap = cv2.VideoCapture(1)
    last_frame = None
    last_time = time.time()

    up = 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if last_frame is not None:
            matches = match(frame, last_frame)
            filtered_matches = filter_outliers(matches, 100)
            scale, theta, tx, ty, l_mean = get_transoform(filtered_matches)
            up += theta

            # Translation
            cv2.line(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)),
                     (int(frame.shape[1] / 2 + tx), int(frame.shape[0] / 2 + ty)), (255, 0, 0), thickness=4)
            cv2.circle(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)), 5, (200, 0, 0), thickness=2)

            # Scale
            s1 = 80
            s2 = frame.shape[0] - 20
            s3 = int((scale - 0.8) / 0.4 * s2 + (1.0 - (scale - 0.8) / 0.4) * s1)
            cv2.line(frame, (40, s1), (40, s2), (0, 0, 200), thickness=2)
            cv2.line(frame, (20, int(s1/2 + s2/2)), (60, int(s1/2 + s2/2)), (0, 0, 200), thickness=2)
            cv2.line(frame, (20, s3), (60, s3), (0, 0, 255), thickness=2)


            # Rotation (up)
            cv2.line(frame, (40, 40),
                     (int(40 + math.sin(-up) * 20), int(40 - math.cos(-up) * 20)), (0, 0, 255), thickness=4)
            cv2.circle(frame, (40, 40), 20, (0, 0, 200), thickness=2)

            debug_img = visualize_matching(frame, filtered_matches)

            # Display the resulting frame
            cv2.imshow('frame', debug_img)

        print("FPS: %.1f" % (1 / (time.time() - last_time)))
        last_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        last_frame = frame

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test()