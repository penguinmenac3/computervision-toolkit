import cv2
import time


stereo = cv2.StereoBM_create(numDisparities=16, blockSize=55)


def depth_map(imgL, imgR):
    return stereo.compute(imgL, imgR)


def test():
    cap_l = cv2.VideoCapture(2)
    cap_r = cv2.VideoCapture(1)
    last_time = time.time()

    while True:
        # Capture frame-by-frame
        ret_l, left = cap_l.read()
        ret_r, right = cap_r.read()

        if ret_l and ret_r:
            left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            disparity = depth_map(left, right)

            # Display the resulting frame
            cv2.imshow('frame', disparity)

        print("FPS: %.1f" % (1 / (time.time() - last_time)))
        last_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test()
