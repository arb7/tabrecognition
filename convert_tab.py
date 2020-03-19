# (c) 2020 Ari Ball-Burack

import sys
import cv2
import argparse
import numpy as np
import scipy.signal
import more_itertools
import tensorflow as tf


def run(fname, line):
    print("Reading {}".format(fname))
    image = cv2.imread(fname)
    print("Preprocessing")
    preprocessed = preprocess(image)
    print("Finding string lines")
    strings = find_string_lines(preprocessed)
    print("Finding intersections")
    intersections = find_intersections(preprocessed, strings)
    if line:
        print("Extracting features with stave lines")
        features = extract_features_line(intersections)
    else:
        print("Extracting features without stave lines")
        features = extract_features_noline(intersections)
    print("Making predictions")
    model_name = 'saved_model/with_lines' if line else 'saved_model/without_lines'
    predictions = get_predictions(features, model_name, line)
    print("Validating")
    validate(predictions, preprocessed, fname, line)
    print("Done")


def preprocess(image):
    """
    Preprocess image by grayscaling, thresholding, opening, edging, detecting
    the page corners, and perspective transforming to give a "scan" of the page

    image: image to preprocess (as sp array)
    return: preprocessed image
    """
    width = 1680
    height = 2376
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 31, 2)
    # cv2.imwrite('img/gray.png', gray) for validation purposes
    opened_gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((7,7)))
    # cv2.imwrite('img/opened_gray.png', opened_gray) for validation purposes

    edged = cv2.Canny(opened_gray, 10, 30)
    edged = cv2.dilate(edged, np.ones((4,4)))
    # cv2.imwrite('img/edged.png', edged) for validation purposes

    cnts, h = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=lambda cnt: cv2.contourArea(cnt), reverse=True)[:5]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    pts2 = np.float32([[0, 0], [0, 2375], [1679, 2375], [1679, 0]])
    pts1 = np.float32(screenCnt)

    min_mean = 2375
    pivot = 0
    for i in range(len(pts1)):
        mean = np.mean(pts1[i])
        if mean < min_mean:
            min_mean = mean
            pivot = i
    pts1_2 = pts1[:pivot]
    pts1_1 = pts1[pivot:]
    pts1 = np.append(pts1_1, pts1_2, 0)

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(gray, M, (1680, 2376))
    dst2 = dst[150:-150, 120:-40]
    # cv2.imwrite('img/dst2.png', dst2) for validation purposes
    opened_dst = cv2.morphologyEx(dst2.copy(), cv2.MORPH_CLOSE, np.ones((3,3)))
    # cv2.imwrite('img/opened_dst.png', opened_dst) for validation purposes
    return opened_dst


def find_string_lines(img):
    """
    Find stave lines (which represent strings) using a histogram of black pixels

    img: np array of preprocessed image
    return: list of y-values of stave lines
    """
    num_rows = img.shape[0]
    row_black_pixel_histogram = []

    erosion = cv2.erode(img, np.ones((5,5)), iterations=1)

    for i in range(num_rows):
        row = erosion[i]
        num_black_pixels = 0
        for j in range(len(row)):
            if (row[j] == 0):
                num_black_pixels += 1

        row_black_pixel_histogram.append(num_black_pixels)

    peaks, _ = scipy.signal.find_peaks(row_black_pixel_histogram, distance=30, prominence=100)
    assert len(peaks) == 36, "Found {} peaks, not 36".format(len(peaks))

    return peaks


def find_intersections(img, peaks):
    """
    Find x-values where the stave lines are intersected (by features)

    img: preprocessed image
    peaks: y-values of stave lines
    return: nested lists -- page list contains staff lists contain string lists,
    which contain lists of tuples of (sub-regions of the preprocessed image that are likely
    to contain a feature (digit) , coordinates of sub-region)
    """
    staff = img.copy()
    staff = cv2.bitwise_not(staff)
    h, w = staff.shape
    buffer_y = 15
    win_size = 30
    char_buffer = 15
    char_buffer_spacing = 18
    border_buffer = 10
    threshold = 1.3

    staff_lines = []
    for i in range(6):
        staff_lines.append(peaks[i*6:6*(i+1)])

    page_intersections = []

    # If the mean value of a small window surpasses a threshold multiple of
    # its surroundings, then it is likely to contain a feature
    for staff_line in staff_lines:
        staff_intersections = []
        for string in staff_line:
            intersections = []
            for x in range(border_buffer,w-2*char_buffer-border_buffer):
                left_border = staff[string-buffer_y:string+buffer_y+1, x-border_buffer:x]
                right_border = staff[string-buffer_y:string+buffer_y+1, x+2*char_buffer:x+2*char_buffer+border_buffer]
                char_whole = staff[string-buffer_y:string+buffer_y+1, x:x+2*char_buffer]
                if np.mean(char_whole) > (threshold*np.mean(left_border+right_border)):
                    intersections.append((x+char_buffer, string))

            # Group together keypoints representing the same digit
            groups = []
            i = 0
            while i < len(intersections):
                count = 0
                group = []

                while True:
                    if i+count == len(intersections):
                        i += count
                        groups.append(group)
                        break
                    elt = intersections[i+count]
                    if not group:
                        group.append(elt)
                        count += 1
                    elif abs(elt[0]-group[-1][0]) < char_buffer_spacing:
                        group.append(elt)
                        count += 1
                    else:
                        i += count
                        groups.append(group)
                        break

            kp = []
            for group in groups:
                g_x = [g[0] for g in group]
                g_mean = (int(np.mean(g_x)), group[0][1])
                kp.append(g_mean)
            staff_intersections.append(kp)

        page_intersections.append(staff_intersections)

    page_subregions = []
    for staff_line in page_intersections:
        staff_subregions = []
        for string_line in staff_line:
            string_subregions = []
            for (x, y) in string_line:
                sub_region = staff.copy()
                window_size = np.min([win_size, x, 1680-x])
                sub_region = sub_region[y-window_size:y+window_size,x-window_size:x+window_size]
                string_subregions.append((sub_region, (x, y)))
            staff_subregions.append(string_subregions)
        page_subregions.append(staff_subregions)

    # Uncomment the below to validate intersection finding
    """flat_list = [item for sublist in page_subregions for item in sublist]
    flat_list = [item for sublist in flat_list for item in sublist]
    inter = img.copy()
    inter = cv2.cvtColor(inter, cv2.COLOR_GRAY2BGR)
    for _, coords in flat_list:
        cv2.circle(inter, coords, 10, (0,0,255), thickness=-1)"""
    # cv2.imwrite('img/intersections.png', inter) for validation purposes
    return page_subregions


def extract_features_noline(page_subregions):
    """
    Extract image and coordinates of features (digits), removing stave lines

    page_subregions: tuples of sub-regions and coordinates
    return: nested lists of tuples of features (cropped with stave line removed)
    and their coordinates
    """
    buffer = 1
    min_width = 2
    w_buf = 3
    int_buf_y = 2
    int_buf_x = 4
    threshold = 150
    counter = 0

    page_features = []
    for staff_subs in page_subregions:
        staff_features = []
        for subs in staff_subs:
            features = []
            for sub_region, (orig_x, orig_y) in subs:
                try:
                    orig_x -= int(sub_region.shape[1]/2)
                    orig_y -= int(sub_region.shape[0]/2)
                    sub = sub_region.copy()
                    sub = sub[w_buf:-w_buf,w_buf:-w_buf]

                    # Find the y-coordinates of the stave line within this sub-region
                    rect = []
                    for x in [0, sub.shape[1]-1]:
                        a = sub[0:sub.shape[0]-1,x]
                        nonzero = [i for i in range(len(a)) if a[i] > 0]
                        while True:
                            if not nonzero or len(nonzero) < min_width+1:
                                break
                            if nonzero[min_width]-nonzero[0] > min_width and nonzero:
                                nonzero.pop(0)
                            else:
                                break
                        while True:
                            if not nonzero or len(nonzero) < min_width+1:
                                break
                            if nonzero[-1]-nonzero[-min_width-1] > min_width and nonzero:
                                nonzero.pop(-1)
                            else:
                                break
                        if nonzero:
                            rect.append((x, nonzero[0]-buffer))
                            rect.append((x, nonzero[-1]+buffer))

                    if len(rect) == 2:
                        x_cur = rect[0][0]
                        x_new = sub.shape[1]-1 if x_cur == 0 else 0
                        rect.append((x_new, rect[0][1]))
                        rect.append((x_new, rect[1][1]))

                    pt1 = min(rect, key=lambda x: x[1])
                    pt2 = max(rect, key=lambda x: x[1])
                    roi_x1, roi_x2, roi_y1, roi_y2 = 0, sub.shape[0]-1, pt1[1], pt2[1]

                    roi = sub[roi_y1:roi_y2, roi_x1:roi_x2]

                    # Zero out the stave line
                    sub[roi_y1:roi_y2, roi_x1:roi_x2] = np.zeros(roi.shape)

                    # Search for, and repopulate with the value 255, places
                    # where the stave line intersected the digit
                    rects = []
                    for y in [(roi_y1, '-'), (roi_y2, '+')]:
                        nonzero = []
                        y_val = y[0]+buffer if y[1]=='+' else y[0]-buffer
                        y_val_2 = y_val+int_buf_y if y[1]=='+' else y_val-int_buf_y
                        for x in range(sub.shape[0]-int_buf_x):
                            candidate = sub[y_val:y_val_2, x:x+int_buf_x] if y[1]=='+' else sub[y_val_2:y_val, x:x+int_buf_x]
                            if np.mean(candidate) > threshold:
                                nonzero.append(x+int(int_buf_x/2))

                        nonzero_grouped = [list(g) for g in more_itertools.consecutive_groups(nonzero)]
                        nonzero = []
                        for g in nonzero_grouped:
                            nonzero.append(g[0])
                            nonzero.append(g[-1])
                        if len(nonzero) == 1:
                            nonzero.append(nonzero[0])
                        while nonzero:
                            rect = [(nonzero[0], y_val), (nonzero[1], y_val)]
                            del nonzero[1]
                            del nonzero[0]
                            rects.append(rect)

                    if not rects:
                        counter += 1
                        continue

                    rects = sorted(rects, key=lambda x: x[0][1])

                    top = []
                    bottom = []

                    top.append(rects.pop(0))
                    while rects:
                        if rects[0][0][1] == top[0][0][1]:
                            top.append(rects.pop(0))
                        else:
                            bottom.append(rects.pop(0))
                    if len(top) != len(bottom):
                        # print('rects {} are mismatched'.format(counter))
                        if not bottom:
                            for t in top:
                                y_cur = t[0][1]
                                y_choices = (roi_y1-buffer, roi_y2+buffer)
                                y_new = y_choices[0] if y_cur == y_choices[1] else y_choices[1]
                                bottom.append([(t[0][0], y_new), (t[1][0], y_new)])
                        while len(top) != len(bottom):
                            longer = max(top, bottom, key=lambda x: len(x))
                            shorter = min(top, bottom, key=lambda x: len(x))

                            if len(longer) == 2 and len(shorter) == 1:
                                long_dist = longer[1][0][0] - longer[0][0][0]
                                long_dist_1 = longer[0][1][0] - longer[0][0][0]
                                long_dist_2 = longer[1][1][0] - longer[1][0][0]
                                min_w = long_dist/2
                                first_x, second_x = shorter[0][0][0], shorter[0][1][0]
                                short_y = shorter[0][0][1]
                                if (second_x - first_x) > min_w:
                                    shorter.clear()
                                    shorter.append([(first_x, short_y), (first_x+long_dist_1, short_y)])
                                    shorter.append([(second_x, short_y), (second_x+long_dist_2, short_y)])
                                    break

                            max_dist = 0
                            max_dist_index = 0
                            for ind in range(len(longer)):
                                (x_pos, _), _ = longer[ind]
                                for (x_pos_2, _), _ in shorter:
                                    dist = abs(x_pos - x_pos_2)
                                    if dist > max_dist:
                                        max_dist = dist
                                        max_dist_index = ind
                            del longer[max_dist_index]
                    rectangles = []
                    for i in range(len(top)):
                        rectangles.append(top[i] + bottom[i])

                    rects = [cv2.convexHull(np.float32(rect)) for rect in rectangles]

                    mask = np.zeros(sub.shape, dtype='int32')
                    for rect in rects:
                        cv2.fillConvexPoly(mask, np.array(rect, 'int32'), 255)

                    sub = sub + mask
                    sub = np.clip(sub, 0, 255)
                    sub = np.float32(sub)

                    # Find the new moment (center) of the image
                    moment = cv2.moments(sub)
                    m_x = int(moment ["m10"] / moment["m00"])
                    m_y = int(moment ["m01"] / moment["m00"])

                    # Intelligently crop the image so the whole digit is
                    # preserved but nothing else
                    prev_tot = 0
                    for size in range(15,30):
                        new_sub = sub[m_y-size:m_y+size, m_x-size:m_x+size]
                        tot = np.sum(new_sub)
                        if (tot - prev_tot) > 255*size:
                            prev_tot = tot
                        else:
                            break
                    size += 3
                    size = min([m_x, m_y, size])
                    sub = sub[m_y-size:m_y+size, m_x-size:m_x+size]

                    # Uncomment the below to validate
                    # cv2.imwrite('img/cropped/cropped_{}.png'.format(counter), sub)

                    final_x = orig_x + m_x
                    final_y = orig_y + m_y

                    features.append((sub, (final_x, final_y)))
                    counter += 1
                except:
                    print('Error on index {}: {}'.format(counter, sys.exc_info()[1]))
                    counter += 1
            staff_features.append(features)
        page_features.append(staff_features)
    return page_features


def extract_features_line(page_subregions):
    """
    Extract image and coordinates of features (digits), preserving stave lines

    page_subregions: tuples of sub-regions and coordinates
    return: nested lists of tuples of features and their coordinates
    """
    buffer = 1
    min_width = 2
    w_buf = 3
    counter = 0

    page_features = []
    for staff_subs in page_subregions:
        staff_features = []
        for subs in staff_subs:
            features = []
            for sub_region, (orig_x, orig_y) in subs:
                try:
                    orig_x -= int(sub_region.shape[1]/2)
                    orig_y -= int(sub_region.shape[0]/2)
                    sub = sub_region.copy()
                    sub = sub[w_buf:-w_buf,w_buf:-w_buf]

                    # Find stave line locations only for intelligent cropping
                    rect = []
                    for x in [0, sub.shape[1]-1]:
                        a = sub[0:sub.shape[0]-1,x]
                        nonzero = [i for i in range(len(a)) if a[i] > 127]
                        while True:
                            if not nonzero:
                                break
                            if nonzero[min_width]-nonzero[0] > min_width:
                                nonzero.pop(0)
                            else:
                                break
                        while True:
                            if not nonzero:
                                break
                            if nonzero[-1]-nonzero[-min_width-1] > min_width:
                                nonzero.pop(-1)
                            else:
                                break
                        if nonzero:
                            rect.append((x, nonzero[0]-buffer))
                            rect.append((x, nonzero[-1]+buffer))

                    if len(rect) == 2:
                        x_cur = rect[0][0]
                        x_new = sub.shape[1]-1 if x_cur == 0 else 0
                        rect.append((x_new, rect[0][1]))
                        rect.append((x_new, rect[1][1]))

                    pt1 = min(rect, key=lambda x: x[1])
                    pt2 = max(rect, key=lambda x: x[1])

                    mean_y = int(np.mean([pt2[1], pt1[1]])+0.5)

                    prev_tot = 0
                    for top_height in range(1,int(sub.shape[0]/2)):
                        new_sub = sub.copy()
                        new_sub = new_sub[mean_y-top_height:new_sub.shape[0], 0:sub.shape[1]]
                        tot = np.sum(new_sub)
                        if tot > prev_tot:
                            prev_tot = tot
                        else:
                            if mean_y < top_height:
                                top_height -= 1
                            break

                    prev_tot = 0
                    for bottom_height in range(1,int(sub.shape[0]/2)):
                        new_sub = sub.copy()
                        new_sub = new_sub[0:mean_y+bottom_height, 0:sub.shape[1]]
                        tot = np.sum(new_sub)
                        if tot > prev_tot:
                            prev_tot = tot
                        else:
                            if (mean_y+bottom_height) > sub.shape[0]:
                                bottom_height -= 1
                            break

                    moment = cv2.moments(sub)
                    m_x = int(moment ["m10"] / moment["m00"])
                    m_y = int(moment ["m01"] / moment["m00"])

                    dim = int((top_height + bottom_height + 1)/2)
                    constr_x = np.min([m_x, (sub.shape[1]-m_x)])
                    constr_y = np.min([mean_y, (sub.shape[0]-m_y)])

                    dim = np.min([dim, constr_x, constr_y])

                    sub = sub[mean_y-top_height:mean_y+bottom_height, m_x-dim:m_x+dim]
                    final_x = orig_x + m_x
                    final_y = orig_y + m_y

                    # Uncomment the below to validate
                    # cv2.imwrite('img/cropped_line/cropped_{}.png'.format(counter), sub)
                    features.append((sub, (final_x, final_y)))
                    counter += 1
                except:
                    print(counter, sys.exc_info()[1])
                    counter += 1
            staff_features.append(features)
        page_features.append(staff_features)
    return page_features


def get_predictions(page_features, model_name, line):
    """
    For each identified feature, predict the value of the digit using the
    appropriately trained model

    page_features: sub-regions representing features, and their coordinates
    model_name: name of the TF model to use
    line: whether stave lines have been preserved, for reporting average line
    width information
    return: predictions and their coordinates
    """
    model = tf.keras.models.load_model(model_name)

    counter = 0
    min_width = 2
    mids = []
    widths = []
    all_preds = []
    for staff_features in page_features:
        staff_preds = []
        for string_features in staff_features:
            preds = []
            for feature, (feature_x, feature_y) in string_features:
                # Convert to grayscale, fill out to be square, resize, and erode
                gray = feature.copy()
                h, w = gray.shape
                diff = h - w
                if diff > 0:
                    gray = cv2.copyMakeBorder(gray, 0, 0, 0, diff, 0)
                gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
                gray = cv2.erode(gray, np.ones((5,5)), iterations=1)

                # If stave lines are present, record line position and width
                if line:
                    for_width = gray.copy()
                    for_width = cv2.resize(for_width, (28, 28), interpolation=cv2.INTER_AREA)

                    rect = []
                    for x in [0, for_width.shape[1]-1]:
                        a = for_width[0:for_width.shape[0]-1,x]
                        nonzero = [i for i in range(len(a)) if a[i] > 0]
                        while True:
                            if len(nonzero) <= min_width:
                                break
                            if nonzero[min_width]-nonzero[0] > min_width:
                                nonzero.pop(0)
                            else:
                                break
                        while True:
                            if len(nonzero) <= min_width:
                                break
                            if nonzero[-1]-nonzero[-min_width-1] > min_width:
                                nonzero.pop(-1)
                            else:
                                break
                        if nonzero:
                            rect.append((x, nonzero[0]))
                            rect.append((x, nonzero[-1]))

                    if len(rect) == 2:
                        x_cur = rect[0][0]
                        x_new = for_width.shape[1]-1 if x_cur == 0 else 0
                        rect.append((x_new, rect[0][1]))
                        rect.append((x_new, rect[1][1]))
                    elif len(rect) == 0:
                        counter += 1
                        continue

                    top = np.mean([rect[0][1], rect[2][1]])
                    bottom = np.mean([rect[1][1], rect[3][1]])
                    width = bottom-top
                    mids.append(np.mean([top, bottom]))
                    widths.append(width/2)

                # Uncomment the below to validate
                # cv2.imwrite('img/for_classifier/for_classifier_{}.png'.format(counter), gray)

                # Resize, normalize to [0, 1], and predict using TF model
                gray = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
                gray = np.divide(gray, 255)
                pred = model.predict(gray.reshape(1, 28, 28, 1))
                preds.append((pred.argmax(), (feature_x, feature_y)))
                counter += 1
            staff_preds.append(preds)
        all_preds.append(staff_preds)

    # Print line width information if appropriate
    if line:
        print(np.mean(mids), np.std(mids))
    return all_preds


def validate(all_preds, img, fname, line):
    """
    Produce a validation image that superposes predictions on the preprocessed
    image

    all_preds: predictions and their coordinates
    img: preprocessed image
    fname: file name of input (non-preprocessed) image
    line: whether or not stave lines have been preserved
    """
    flat_list = [item for sublist in all_preds for item in sublist]
    flat_list = [item for sublist in flat_list for item in sublist]

    validate = img.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (0, 0, 255)
    thickness = 4

    validate = cv2.cvtColor(validate, cv2.COLOR_GRAY2BGR)

    for num, (x, y) in flat_list:
        coords = (x+10, y-10) if (x+15) < validate.shape[1] else (x-20, y-10)
        validate = cv2.putText(validate, str(num), coords, font, fontScale, color, thickness)

    imfile = ''.join(fname.split('.')[:-1])
    imfile = '{}_validation_{}.png'.format(imfile, str(line))
    cv2.imwrite(imfile, validate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('imagefile', type=str,
                        help="Image file to convert")
    parser.add_argument('-l', '--lines', action='store_true',
                        help="Train model with artificial stave lines")
    args = parser.parse_args()
    run(args.imagefile, args.lines)
