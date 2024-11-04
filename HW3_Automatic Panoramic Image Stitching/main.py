import argparse
import numpy as np
import cv2
from tqdm import tqdm

class AutomaticPanoramic:

    def __init__(self, args):

        self.data1 = args.data1
        self.data2 = args.data2
        self.save = args.save
        self.ratio = args.ratio
        

    def run(self):

        img1 = cv2.imread(self.data1)
        img2 = cv2.imread(self.data2)

        # Step1 - Interest points detection & feature description by SIFT
        keypoint1, features1 = self.extract_keypoints_and_features(img1)
        keypoint2, features2 = self.extract_keypoints_and_features(img2)

        # Step2 - Feature matching by SIFT features
        matches = self.feature_match(features1, features2, self.ratio)
        matched_img = self.visualize(img1, img2, keypoint1, keypoint2, matches)
        if self.save:
            cv2.imwrite(f"./output/{self.data1.split('/')[-1].split('.')[0]}_matches.jpg", matched_img)
        # plt.imshow(matched_img),plt.show()

        # Step3 - RANSAC to find homography matrix H
        H =  self.homomat(keypoint1, keypoint2, matches)
        # H = self.ransac_api(keypoint1, keypoint2, matches, 5.0)


        # Step4 - Warp image to create panoramic image
        # result = self.warp(img1, img2, H)
        result = self.warp_api(img1, img2, H)
        if self.save:
            cv2.imwrite(f"./output/{self.data1.split('/')[-1].split('.')[0]}_result.jpg", result)
        


    ''' Step1 - Interest points detection & feature description by SIFT '''
    def extract_keypoints_and_features(self, img):

        sift_detector = cv2.SIFT_create()
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        key_pts, features = sift_detector.detectAndCompute(grayscale, None)
        points = np.float32([kp.pt for kp in key_pts])
        return points, features
    

    ''' Step2 - Feature matching by SIFT features '''
    def feature_match(self, f1, f2, ratio):
        distances, raw_matches = [], []
        for idx in range(f1.shape[0]):
            closest, second_closest = float('inf'), float('inf')
            match1, match2 = -1, -1
            for i, f in enumerate(f2):
                dist = np.linalg.norm(f1[idx] - f)
                if dist < closest:
                    second_closest, match2 = closest, match1
                    closest, match1 = dist, i
                elif dist < second_closest:
                    second_closest, match2 = dist, i
            distances.append((closest, second_closest))
            raw_matches.append((match1, match2))

        matches = [
            (raw_matches[i][0], i) for i, (closest, second_closest) in enumerate(distances)
            if closest < ratio * second_closest
        ]
        
        return matches

    def visualize(self, img1, img2, points1, points2, matches):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        combined_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        combined_img[0:h1, 0:w1], combined_img[0:h2, w1:] = img1, img2

        for (idx2, idx1) in matches:
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            pt1, pt2 = (int(points1[idx1][0]), int(points1[idx1][1])), (int(points2[idx2][0]) + w1, int(points2[idx2][1]))
            cv2.line(combined_img, pt1, pt2, color, 1)
        
        return combined_img


    ''' Step3 - RANSAC to find homography matrix H '''
    def homomat(self, points_in_img1, points_in_img2, matches, sample=4, iteration=2000, threshold=5.0):
        points_in_img1 = np.float32([points_in_img1[i] for (_,i) in matches])
        points_in_img2 = np.float32([points_in_img2[i] for (i,_) in matches])


        num_points = points_in_img1.shape[0]
        best_H = None
        best_inliers = 0
        for _ in tqdm(range(iteration), desc="Computing homography"):

            idx = np.random.choice(points_in_img1.shape[0], sample, replace=False)
            sampled_points1 = points_in_img1[idx]
            sampled_points2 = points_in_img2[idx]

            H = self.compute_homography(sampled_points1, sampled_points2) # H (3,3)

            inliers = 0
            for i in range(num_points):
                pt1 = np.append(points_in_img1[i], 1)  # homogeneous coordinates
                projected_pt = H @ pt1
                projected_pt /= projected_pt[2]

                # Calculate error
                pt2 = points_in_img2[i]
                error = np.linalg.norm(pt2 - projected_pt[:2])

                if error < threshold:
                    inliers += 1

            if inliers > best_inliers:
                best_inliers = inliers
                best_H = H

        print(f"Best H: {best_H}")
            
        return best_H

    def compute_homography(self, points1, points2):
        
        A = []
        for i in range(points1.shape[0]):
            x1, y1 = points1[i]
            x2, y2 = points2[i]
            A.append([-x1, -y1, -1, 0, 0, 0, x1*x2, y1*x2, x2])
            A.append([0, 0, 0, -x1, -y1, -1, x1*y2, y1*y2, y2])
        A = np.array(A)

        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        H = H / H[2, 2]

        return H

    def ransac_api(self, src_pts, dst_pts, matches,sample=50, iteration=3, threshold=5.0):
        p1 = np.float32([src_pts[i] for (_,i) in matches])
        p2 = np.float32([dst_pts[i] for (i,_) in matches])
        H, _ = cv2.findHomography(p1, p2, cv2.RANSAC, threshold)

        print(f"ransac_api H : \n{H}")
        return H


    '''Step4 - Warp image to create panoramic image'''
    def warp(self):
        pass

    def warp_api(self, img1, img2, H):
        total_width = img1.shape[1] + img2.shape[1]
        warp_result = cv2.warpPerspective(img1, H, (total_width , img1.shape[0]))
        warp_result[0:img2.shape[0], 0:img2.shape[1]] = img2
        return warp_result


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('-d1', '--data1', default= "./data/hill2.JPG", type= str, help= 'path of image1')
    parse.add_argument('-d2', '--data2', default= "./data/hill1.JPG", type= str, help= 'path of image2')
    parse.add_argument('-s', '--save', default= False, type= bool, help= 'save image')
    parse.add_argument('-rl', '--ratio', default= 0.7, type= float, help= '')

    args = parse.parse_args()

    return args

if __name__ == "__main__":

    args = parse_args()
    automatic_panoramic = AutomaticPanoramic(args)
    automatic_panoramic.run()