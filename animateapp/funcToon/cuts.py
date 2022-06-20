import cv2
import numpy as np

from tensorflow.python.keras.models import load_model
IMAGE_SIZE = 224

# 컷 모델
model = load_model('model/best_gray_model_2.h5')
# 말풍선 모델
model_b = load_model('model/bubble_gray_model.h5')


def image_preproc(img_list):  # 이 코드는 전처리부분만을 가져왔습니다.
    img_input = []

    # 모델 입력 전처리
    for img in img_list:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray_res = cv2.resize(img_gray, dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        img_input.append(img_gray_res / 255)
    img_input = np.asarray(img_input)
    img_input = img_input.reshape(img_input.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1)

    # 모델 적용 컷 or 말풍선
    cut_predict = model.predict(img_input).reshape(len(img_input), IMAGE_SIZE, IMAGE_SIZE)
    bubble_predict = model_b.predict(img_input).reshape(len(img_input), IMAGE_SIZE, IMAGE_SIZE)

    # 출력 이미지 전처리
    # 이제 컷과 말풍선을 동시에 처리합니다.
    labels_list = []
    for predict in [cut_predict, bubble_predict]:
        labels = [np.around(label) * 255 for label in predict]
        # 여기서 다시 출력이미지 사이즈를 키움
        labels = [cv2.resize(label, dsize=img_gray.shape[::-1], interpolation=cv2.INTER_AREA) for label in labels]
        labels_list.append(labels)

    return labels_list[0], labels_list[1]  # 0 : cut, 1 : bubble


def split_cut_b(img, polygon, page_num, is_bubble=False):
    x, y, w, h = cv2.boundingRect(polygon)  # 폴리곤으로 bounding 박스 그림
    croped = img[y:y + h, x:x + w].copy()  # 원본 이미지에서 자름.

    # 컷일 경우에만 패딩을 추가합니다.
    if not is_bubble:
        # ******* 현 패딩 > 컷의 높이와 너비 중 큰 값의 0.45
        # ******* 대부분의 컷은 0.3정도로도 잘 작동하는데 어떤 컷은 0.4에도 터집니다.
        # ******* ex) 2부 4화 94p
        padding = int(max(h, w) * 0.45)
        startline = int(padding / 2)

        img_padding = np.zeros([h + padding, w + padding, 3], np.uint8)
        img_padding[startline:startline + h, startline: startline + w] = croped

        croped = img_padding

        pts = polygon - polygon.min(axis=0) + startline
    else:
        pts = polygon - polygon.min(axis=0)

    mask = np.zeros([croped.shape[0], croped.shape[1]], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    dst = cv2.bitwise_and(croped, croped, mask=mask)  # 빈 공간에 컷만 추가.

    # 변경점. 이제 모든 이미지는 x, y, w, h, mask, matching_num 값을 포함합니다.
    # 컷에 말풍선이 없거나 말풍선이 잘못 검출 된 경우를 위해 여기서 초기화합니다.
    if is_bubble:
        txt_per_bub = find_txt_cnt(dst, mask)
        return {'image': dst, 'xywh': [x, y, w, h], 'mask': mask, 'matching_cut_num': -1, 'txt_per_bub': txt_per_bub}
    else:
        return {'image': dst, 'xywh': [x, y, w, h], 'mask': mask, 'matching_bub_num': [], 'padding': padding}


def find_txt_cnt(bubble, mask):
    _, thresh_mask = cv2.threshold(mask, 127, 1, 0) # 이진화
    bubble_px_cnt = np.sum(thresh_mask)
    bg = np.ones_like(mask)
    res = bubble.copy()
    cv2.bitwise_not(res, res, mask=bg)
    res_gray = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY) # 이미지 1채널로 변경
    _, res_bit = cv2.threshold(res_gray, 127, 1, 0)  ##  이진화
    only_text = cv2.bitwise_and(thresh_mask, thresh_mask, mask=res_bit)
    bubble_txt_cnt = np.sum(only_text)
    txt_per_bub = bubble_txt_cnt / bubble_px_cnt
    return txt_per_bub
# ===========================================================================================================


def sort_cut_b(img_list, contours, read, is_bubble=False):
    height = img_list[0].shape[0]
    width = img_list[0].shape[1]
    n = 0
    centroids = []
    bubble_centers = []
    for contour in contours:
        cx = contour.min(axis=0)[0][0] // int(width / 5)  # 이것은 순서를 정하기 위해서 뽑습니다.
        cy = contour.min(axis=0)[0][1] // int(height / 5)
        centroids.append([cx, cy, n])
        n += 1
    if read == 'L':
        centroids.sort(key=lambda x: (x[1], x[0]))
    elif read == 'R':
        centroids.sort(key=lambda x: (x[1], -x[0]))
    centroids = np.array(centroids)
    index = centroids[:, 2].tolist()
    sort_contours = [contours[i] for i in index]
    if is_bubble:
        for contour in sort_contours:
            centroid = cv2.moments(contour)
            bx = int(centroid['m10'] / centroid['m00'])  # 이건 말풍선과 컷을 매칭하기 위해서 뽑습니다.
            by = int(centroid['m01'] / centroid['m00'])
            bubble_centers.append([bx, by, n])
        return sort_contours, centroids, bubble_centers  # 이걸 이용해 컷과 매칭합니다.
    return sort_contours, centroids


def make_cut_bubble(img_list, labels, read, is_bubble=False):
    cuts_list = []
    centroids_list = []
    polygons_list = []
    bubble_centers_list = []

    for idx, label in enumerate(labels):
        cuts = []
        label = np.asarray(label, dtype=np.uint8)

        contours, hierarchy = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # countour를 찾음.

        # 컷에 말풍선이 하나도 없을 때의 탈출조건
        if is_bubble and (len(contours) == 0):
            cuts_list.append([])
            bubble_centers_list.append([])
            continue

        # 컷 정렬
        if is_bubble:
            contours, centroids, bubble_centers = sort_cut_b(img_list, contours, read, is_bubble=True)
        else:
            contours, centroids = sort_cut_b(img_list, contours, read)

        polygons = [contour.reshape(contour.shape[0], 2) for contour in contours]  # contour reshape

        if is_bubble:
            for polygon in polygons:
                cuts.append(split_cut_b(img_list[idx], polygon, page_num=idx, is_bubble=True))

            bubble_centers_list.append(bubble_centers)
        else:
            for polygon in polygons:
                cuts.append(split_cut_b(img_list[idx], polygon, page_num=idx))

            polygons_list.append(polygons)

        cuts_list.append(cuts)
        centroids_list.append(centroids)

    # 변경점. 이제 모든 출력값은 페이지마다로 구분됩니다.
    # [0] : 페이지0번, [0][0] : 페이지0번의 0번 째 컷 or 버블의 데이터
    if is_bubble:
        return cuts_list, centroids_list, bubble_centers_list  # bubble_centers 는 말풍선 컷 매칭을 위해 뽑습니다.
    else:
        return cuts_list, centroids_list, polygons_list  # 폴리곤은 말풍선 위치결정을 위해서 뽑습니다.
