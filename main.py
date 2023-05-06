import cv2
import numpy as np
import os
from skimage.filters import threshold_sauvola


class StockTinder:
    def __init__(self):
        # current drawing mode
        self.fg_model = None
        self.bg_model = None
        self.drawing = False
        self.x, self.y = 0, 0
        self.mode = 0

        # load file
        files = os.listdir('samples')
        print(files)
        self.img = cv2.imread(f'samples/{files[8]}')
        self.img = cv2.resize(self.img, (int(self.img.shape[1] * 0.2), int(self.img.shape[0] * 0.2)), cv2.INTER_AREA)

        # define models from image size
        self.fg_mask = np.zeros(self.img.shape[:2], dtype='uint8')
        self.bg_mask = np.zeros(self.img.shape[:2], dtype='uint8')
        self.mask = np.zeros(self.img.shape[:2], dtype='uint8')
        self.mask_img = np.zeros(self.img.shape[:2], dtype='uint8')

        self.init_grabcut()

        cv2.namedWindow('Stock Tinder')
        cv2.namedWindow('debug')
        cv2.setMouseCallback('Stock Tinder', self.line_draw_callback)

        while True:
            mask_img = np.where((self.mask_img == 2) | (self.mask_img == 0), 0, 255).astype(
                "uint8"
            )
            # add GUI elements to window

            # clean up mask image slightly
            mask_img = cv2.erode(mask_img, None, iterations=2)
            mask_img = cv2.dilate(mask_img, None, iterations=2)
            mask_img = cv2.dilate(mask_img, None, iterations=2)
            mask_img = cv2.erode(mask_img, None, iterations=2)

            gui_img = np.copy(self.img)
            # mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
            mask_img = cv2.applyColorMap(mask_img, cv2.COLORMAP_COOL)

            cv2.imshow('debug', mask_img)

            gui_img = cv2.addWeighted(gui_img, 0.4, mask_img, 0.2, 0.0, gui_img)

            gui_img[self.fg_mask == 255] = (0, 255, 0)
            gui_img[self.bg_mask == 255] = (0, 0, 255)
            # gui_img[mask_img == 1] = (255, 0, 0)
            cv2.imshow('Stock Tinder', gui_img)

            # keyboard events
            key = cv2.waitKey(1)
            if key & 0xFF == 27:
                break
            elif key == ord('f'):
                self.mode = 0
            elif key == ord('b'):
                self.mode = 1
            elif key == ord('s'):
                # segment and overlay segmentation on image
                self.refine_grabcut(self.fg_mask, self.bg_mask)
                # self.fg_mask = np.zeros(self.img.shape[:2], dtype='uint8')
                # self.bg_mask = np.zeros(self.img.shape[:2], dtype='uint8')
            elif key == ord('r'):
                self.fg_mask = np.zeros(self.img.shape[:2], dtype='uint8')
                self.bg_mask = np.zeros(self.img.shape[:2], dtype='uint8')
                self.init_grabcut()

        cv2.destroyAllWindows()

    def init_grabcut(self):
        print('initializing mask')
        self.mask = self.initial_segmentation(self.img)
        self.mask[self.mask == 255] = cv2.GC_PR_FGD
        self.mask[self.mask == 0] = cv2.GC_PR_BGD

        self.bg_model = np.zeros((1, 65), np.float64)
        self.fg_model = np.zeros((1, 65), np.float64)

        if cv2.countNonZero(self.mask):
            self.mask_img, bg_model, fg_model = cv2.grabCut(self.img, self.mask, None, self.bg_model, self.fg_model, 5, cv2.GC_INIT_WITH_MASK)
            print('done initializing mask')

    def refine_grabcut(self, fg_mask, bg_mask):
        print("thinking...")
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # initialize with rectangle the size of image
        # rect = (0, 0, img.shape[0], img.shape[1])
        # grabcut_mask, bg_model, fg_model = cv2.grabCut(img, mask, rect, bg_model, fg_model, 1, cv2.GC_INIT_WITH_RECT)

        print(fg_mask)
        print(cv2.countNonZero(fg_mask))
        # fine-tune with hand-drawn mask
        self.mask[fg_mask == 255] = cv2.GC_FGD
        self.mask[bg_mask == 255] = cv2.GC_BGD
        self.mask_img, self.bg_model, self.fg_model = cv2.grabCut(self.img, self.mask, None, self.bg_model, self.fg_model, 5, cv2.GC_INIT_WITH_MASK)

        print("done!")

    def line_draw_callback(self, event, origin_x, origin_y, flags, param):
        mask = self.bg_mask if self.mode else self.fg_mask

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.x, self.y = origin_x, origin_y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                cv2.line(mask, (self.x, self.y), (origin_x, origin_y), color=255, thickness=3)
                self.x, self.y = origin_x, origin_y

        elif event == cv2.EVENT_LBUTTONUP:
            cv2.line(mask, (self.x, self.y), (origin_x, origin_y), color=255, thickness=3)
            self.drawing = False

    def initial_segmentation(self, img):
        # img_blur = cv2.bilateralFilter(src=img, d=9, sigmaColor=100, sigmaSpace=100)
        img_blur = cv2.GaussianBlur(img, (5, 5), 0)
        # img_blur = cv2.medianBlur(img_blur, 3)

        img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
        # ret, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # ret, img_thresh = cv2.threshold(img_gray, 140, 255, cv2.THRESH_BINARY)

        img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        img_thresh = cv2.bitwise_not(img_thresh)

        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        # img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
        # img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)

        # get background color as median of everything outside largest contour
        # get stock color as median of everything inside largest contour
        # largest_contour = max(contours, key=cv2.contourArea)
        # cv2.drawContours(img_thresh, [largest_contour], -1, 255, 1)
        #
        # for contour in contours:
        #     if cv2.contourArea(contour) < 2000:
        #         # classify each contour as outer or inner contour
        #         cv2.drawContours(img_thresh, [contour], 0, 1, -1)

        img_thresh = cv2.dilate(img_thresh, None, iterations=2)
        img_thresh = cv2.erode(img_thresh, None, iterations=2)

        # ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # img = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
        # img = cv2.Canny(image=img, threshold1=10, threshold2=200)
        #
        # from skimage.filters import sobel
        # img_sobel = sobel(img_gray)
        # img_sobel = ((img_sobel / img_sobel.max()) * 255).astype('uint8')

        # ret, img_thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)

        # img_thresh = cv2.erode(img_sobel, None, iterations=2)
        # img_thresh = cv2.dilate(img_sobel, None, iterations=2)

        # def fill_hole(input_mask):
        #     h, w = input_mask.shape
        #     canvas = np.zeros((h + 2, w + 2), np.uint8)
        #     canvas[1:h + 1, 1:w + 1] = input_mask.copy()
        #     mask = np.zeros((h + 4, w + 4), np.uint8)
        #     cv2.floodFill(canvas, mask, (0, 0), 1)
        #     canvas = canvas[1:h + 1, 1:w + 1].astype(np.bool)
        #
        #     return ~canvas | input_mask

        # img_thresh = cv2.bitwise_not(img_thresh)

        # gray = cv2.bitwise_not(img_thresh)

        contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour for contour in contours if cv2.contourArea(contour) > 1000]
        # contours = [max(contours, key=cv2.contourArea)]

        mask = np.zeros(self.img.shape[:2], dtype='uint8')
        # largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)

        return mask

if __name__ == '__main__':
    stock_tinder = StockTinder()
    # cv2.imwrite('output.png', mask)