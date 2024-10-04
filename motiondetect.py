import cv2

# Initialize variables
static_back = None
motion_list = [None, None]
time = []
video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    motion = 0

    # Convert frame to grayscale and blur it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Initialize static background
    if static_back is None:
        static_back = gray
        continue

    # Update the background gradually
    static_back = cv2.addWeighted(static_back, 0.95, gray, 0.05, 0)

    # Compute the absolute difference
    diff_frame = cv2.absdiff(static_back, gray)

    # Apply adaptive threshold
    thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # Find contours
    cnts, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue
        motion = 1

        # Draw bounding rectangle
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Display frames
    cv2.imshow("Gray Frame", gray)
    cv2.imshow("Difference Frame", diff_frame)
    cv2.imshow("Threshold Frame", thresh_frame)
    cv2.imshow("Color Frame", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
